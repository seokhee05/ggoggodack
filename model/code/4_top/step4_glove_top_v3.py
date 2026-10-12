"""Step 4 V3: separate the white armrest core from plastic, for VISUAL REVIEW.

Run from C:\\ggo (same existing weights as V1):
    python .\\step4_glove_top_v2.py

White surface tracking is only a review candidate, NOT proof of top or inspection.
The color cue is tuned for the WHITE armrest in crop008; visually verify before reuse.
A red side mask can contain both the actual bright surface AND hanging plastic.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
GREEN = (40, 215, 55)
YELLOW = (0, 210, 255)
RED = (65, 65, 245)
CYAN = (255, 245, 55)
GRAY = (100, 100, 100)
BLUE = (245, 185, 40)  # BGR turquoise: white surface review only
PURPLE = (205, 90, 230)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    w = max(0., min(ax2, bx2) - max(ax1, bx1))
    h = max(0., min(ay2, by2) - max(ay1, by1))
    union = max(0., ax2-ax1)*max(0., ay2-ay1) + max(0., bx2-bx1)*max(0., by2-by1) - w*h
    return w*h/max(union, 1.)


def shape_metrics(mask):
    """Geometric cues only: they do not establish that this surface is top."""
    cts, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cts:
        return {'solidity':0., 'rectangularity':0., 'aspect':0., 'compactness':999.}
    c = max(cts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 10:
        return {'solidity':0., 'rectangularity':0., 'aspect':0., 'compactness':999.}
    hull = cv2.convexHull(c)
    hull_area = max(cv2.contourArea(hull), 1.)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    rect_area = max(w*h, 1.)
    perimeter = cv2.arcLength(c, True)
    return {
        'solidity':float(area/hull_area),
        'rectangularity':float(min(1.,area/rect_area)),
        'aspect':float(max(w,h)/max(min(w,h),1.)),
        'compactness':float(perimeter*perimeter/max(4*math.pi*area,1.)),
    }


def rigid_shape(stats):
    """Reject many dangling, irregular plastic masks; thresholds need field checks."""
    return (stats['solidity'] >= .83 and
            stats['rectangularity'] >= .48 and
            stats['aspect'] >= 1.32 and
            stats['compactness'] <= 44.)


def choose_side(sides, anchor, frame_no, fps, max_age=.65):
    """Only propose a temporally associated side mask; never call it TOP."""
    if anchor is None or (frame_no-anchor['frame'])/fps > max_age:
        return None, None
    options=[]
    for det in sides:
        if det['conf'] < .35 or not (.03 <= det['area'] <= .42):
            continue
        if not (.44*anchor['area'] <= det['area'] <= 1.65*anchor['area']):
            continue
        ov = box_iou(det['bbox'], anchor['bbox'])
        if ov < .26:
            continue
        shp = shape_metrics(det['mask'])
        # Ranking prefers similar-object overlap, then shape; threshold gates below.
        score = ov * det['conf'] * (1.0 if rigid_shape(shp) else .35)
        options.append((score, det, shp, ov))
    if not options:
        return None, None
    _, winner, metrics, overlap = max(options, key=lambda x:x[0])
    winner = {**winner, 'shape':metrics, 'anchor_iou':overlap}
    return (winner, 'SIDE_REVIEW') if rigid_shape(metrics) else (winner, 'PLASTIC_RISK')


def get_segments(result, width, height):
    if result.masks is None or result.boxes is None:
        return []
    # Ultralytics masks.xy is in original frame coordinates, unlike masks.data.
    polygons=result.masks.xy
    found=[]
    for i, box in enumerate(result.boxes):
        if i >= len(polygons):
            break
        p=np.asarray(polygons[i],dtype='float32')
        if p.ndim != 2 or len(p) < 3:
            continue
        mask=np.zeros((height,width),dtype='uint8')
        poly=np.rint(p).astype('int32')
        poly[:,0]=np.clip(poly[:,0],0,width-1)
        poly[:,1]=np.clip(poly[:,1],0,height-1)
        cv2.fillPoly(mask,[poly],1)
        area=float(mask.mean())
        found.append({'cls':int(box.cls[0]),'conf':float(box.conf[0]),
                      'bbox':tuple(map(float,box.xyxy[0].tolist())),
                      'area':area,'mask':mask.astype(bool)})
    return found


def draw_mask(frame, mask, color, alpha=.27):
    frame[mask]=(frame[mask]*(1.-alpha)+np.asarray(color)*alpha).astype('uint8')
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame,contours,-1,color,3)


def find_pair(boxes, w, h, prior=None, elapsed=0.):
    diag=math.hypot(w,h)
    detections=[]
    for b in boxes:
        c=float(b.conf[0])
        x1,y1,x2,y2=map(float,b.xyxy[0].tolist())
        area=max(0.,x2-x1)*max(0.,y2-y1)/(w*h)
        cx,cy=(x1+x2)/2,(y1+y2)/2
        if c < .35 or not (.00012 <= area <= .04) or not (.12*h <= cy <= .95*h):
            continue
        detections.append({'conf':c,'bbox':(x1,y1,x2,y2),'center':(cx,cy)})
    pairs=[]
    for i,a in enumerate(detections):
        for b in detections[i+1:]:
            left,right=sorted((a,b),key=lambda x:x['center'][0])
            gap=math.dist(left['center'],right['center'])/diag
            if not (.065 <= gap <= .72):
                continue
            score=left['conf']+right['conf']
            if prior is not None and elapsed <= .32:
                movement=(math.dist(left['center'],prior[0])+math.dist(right['center'],prior[1]))/diag
                # Sudden identity switches/false glove boxes cannot create a 'rising gap'.
                if movement > (.18 + .40*elapsed):
                    continue
                score-=2.*movement
            pairs.append((score,left,right,gap))
    return detections, max(pairs,key=lambda x:x[0]) if pairs else None


def put(frame,text,x,y,color=(245,245,245),scale=.73):
    cv2.putText(frame,text,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)




def white_surface_review(raw, anchor, width, height, *, v_min=215, s_max=60):
    """Propose a *bright connected surface* near previously observed TOP.

    It is deliberately not called top: gloves, white net and packaging can be white.
    No seed TOP => no proposal. Temporal seed expiration is enforced by caller.
    """
    if anchor is None:
        return None
    x1,y1,x2,y2=anchor['bbox']
    dx=max(45,int((x2-x1)*.14));dy=max(40,int((y2-y1)*.19))
    x1=max(0,int(x1)-dx);y1=max(105,int(y1)-dy)
    x2=min(width,int(x2)+dx);y2=min(height-70,int(y2)+dy)
    if x2<=x1 or y2<=y1:
        return None
    hsv=cv2.cvtColor(raw[y1:y2,x1:x2],cv2.COLOR_BGR2HSV)
    bw=((hsv[:,:,2]>=v_min)&(hsv[:,:,1]<=s_max)).astype('uint8')
    bw=cv2.morphologyEx(bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    bw=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    contours,_=cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    proposals=[]
    for c in sorted(contours,key=cv2.contourArea,reverse=True)[:5]:
        area=cv2.contourArea(c)
        ratio=area/(width*height)
        if not .042 <= ratio <= .32:
            continue
        local=np.zeros_like(bw)
        cv2.drawContours(local,[c],-1,1,-1)
        # Full-frame bbox and binary mask; never count the rectangular ROI as a mask.
        bx,by,bww,bhh=cv2.boundingRect(c)
        box=(float(x1+bx),float(y1+by),float(x1+bx+bww),float(y1+by+bhh))
        ov=box_iou(box,anchor['bbox'])
        if ov<.14:
            continue
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        ax,ay=(anchor['bbox'][0]+anchor['bbox'][2])/2,(anchor['bbox'][1]+anchor['bbox'][3])/2
        if math.hypot(cx-ax,cy-ay)/math.hypot(width,height) > .19:
            continue
        hull=max(cv2.contourArea(cv2.convexHull(c)),1.)
        solidity=area/hull
        (_, _),(rw,rh),_=cv2.minAreaRect(c)
        aspect=max(rw,rh)/max(min(rw,rh),1.)
        if solidity<.72 or aspect<1.25:
            continue
        mask=np.zeros((height,width),dtype='bool')
        mask[y1:y2,x1:x2]=local.astype(bool)
        proposals.append({'mask':mask,'bbox':box,'area':float(mask.mean()),
                          'solidity':round(solidity,3),'anchor_iou':round(ov,3),
                          'aspect':round(aspect,3),'score':ov+ratio})
    return max(proposals,key=lambda x:x['score'],default=None)


def main():
    p=argparse.ArgumentParser(description='V3: keep bright top candidates separate from plastic-risk side masks')
    p.add_argument('--video',type=Path,default=ROOT/'crop008.mov')
    p.add_argument('--seg-model',type=Path,default=ROOT/'armrest_3class_best.pt')
    p.add_argument('--glove-model',type=Path,default=ROOT/'glove_runs'/'glove_detector_v2'/'weights'/'best.pt')
    p.add_argument('--start',type=float,default=11.)
    p.add_argument('--end',type=float,default=15.)
    p.add_argument('--white-v',type=int,default=215,help='WHITE REVIEW only: HSV brightness min')
    p.add_argument('--white-s',type=int,default=60,help='WHITE REVIEW only: HSV saturation max')
    p.add_argument('--out',type=Path,default=ROOT/'step4_fusion_v3')
    args=p.parse_args()
    for name,path in [('video',args.video),('seg-model',args.seg_model),('glove-model',args.glove_model)]:
        if not path.is_file():raise FileNotFoundError(f'{name} file missing: {path}')
    if args.start<0 or args.end<=args.start:raise ValueError('check --start and --end')
    from ultralytics import YOLO
    segmenter=YOLO(str(args.seg_model));gloves=YOLO(str(args.glove_model))
    names={int(k):str(v) for k,v in segmenter.names.items()}
    top_id=next((k for k,v in names.items() if v=='top'),None)
    side_id=next((k for k,v in names.items() if v=='side'),None)
    if top_id is None or side_id is None:raise ValueError(f'Expected top and side: {names}')
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise RuntimeError(f'Cannot read {args.video}')
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 30)
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start=round(args.start*fps);end=min(round(args.end*fps),total_n)
    if end<=start:raise ValueError('Empty video range')
    args.out.mkdir(parents=True,exist_ok=True)
    stem=f'{args.video.stem}_step4_v3_{args.start:g}_{args.end:g}s'
    video_path=args.out/(stem+'.mp4');csv_path=args.out/(stem+'.csv')
    writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise RuntimeError(f'Cannot write {video_path}')
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)

    anchor=None;last_seed=None;glove_prior=None;last_glove=None
    gap_history=deque();history=[];glove_flags=[];rise_flags=[]
    counts={'TOP_DIRECT':0,'WHITE_REVIEW':0,'SIDE_REVIEW':0,'PLASTIC_RISK':0,'NO_TOP_EVIDENCE':0,'MIXED_REVIEW_RISK':0}
    max_gap=0.;num_pairs=0;num_rise=0;processed=0
    try:
        with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
            rows=csv.writer(f)
            rows.writerow(['frame_1based','time_sec','display_status','top_direct','white_review',
                           'side_review','plastic_risk','top_conf','side_conf',
                           'white_area_ratio','white_anchor_iou','white_solidity',
                           'two_gloves','gap_px','gap_norm','gap_rise',
                           'direct_top_observed_sec','white_review_sec'])
            for idx in range(start,end):
                ok,raw=cap.read()
                if not ok:break
                frame=raw.copy();t=idx/fps;processed+=1
                sr=segmenter.predict(raw,imgsz=512,conf=.20,device='cpu',retina_masks=True,verbose=False)[0]
                found=get_segments(sr,w,h)
                direct=max((d for d in found if d['cls']==top_id and d['conf']>=.4 and d['area']>=.03),
                           key=lambda d:d['conf'],default=None)
                sides=[d for d in found if d['cls']==side_id]
                side=None;side_kind=None;color=None
                if direct is not None:
                    anchor={**direct,'frame':idx}
                    last_seed=idx
                    draw_mask(frame,direct['mask'],GREEN,.26)
                    status='TOP_DIRECT'
                else:
                    if last_seed is not None and (idx-last_seed)/fps<=.85:
                        color=white_surface_review(raw,anchor,w,h,v_min=args.white_v,s_max=args.white_s)
                        side,side_kind=choose_side(sides,anchor,idx,fps,max_age=.65)
                    if color is not None:
                        # A white patch on a previously seen object: REVIEW only, never direct TOP.
                        draw_mask(frame,color['mask'],BLUE,.19)
                        anchor={**color,'frame':idx}
                        if side_kind=='PLASTIC_RISK':
                            # Do NOT tint the white surface red. Red only marks the remaining side/plastic.
                            rem=side['mask'] & (~color['mask'])
                            if rem.any():draw_mask(frame,rem,RED,.09)
                            status='MIXED_REVIEW_RISK'
                        else:
                            status='WHITE_REVIEW'
                    elif side_kind=='SIDE_REVIEW':
                        draw_mask(frame,side['mask'],YELLOW,.14);status='SIDE_REVIEW'
                    elif side_kind=='PLASTIC_RISK':
                        draw_mask(frame,side['mask'],RED,.09);status='PLASTIC_RISK'
                    else:
                        status='NO_TOP_EVIDENCE'
                counts[status]+=1;history.append(status)

                gr=gloves.predict(raw,imgsz=640,conf=.25,device='cpu',verbose=False)[0]
                elapsed=(idx-last_glove)/fps if last_glove is not None else 1e9
                previous=glove_prior if elapsed<=.32 else None
                all_gloves,pair=find_pair(gr.boxes,w,h,previous,elapsed)
                for gl in all_gloves:
                    a,b,c,d=map(int,gl['bbox']);cv2.rectangle(frame,(a,b),(c,d),(200,125,125),1)
                two=pair is not None;gap_px=None;gap_norm=None;cue=False
                if two:
                    num_pairs+=1
                    _,left,right,gap_norm=pair
                    gap_px=math.dist(left['center'],right['center']);max_gap=max(gap_norm,max_gap)
                    if previous is None:gap_history.clear()
                    while gap_history and (idx-gap_history[0][0])/fps>.5:gap_history.popleft()
                    if len(gap_history)>=3 and gap_norm-min(v for _,v in gap_history)>=.06:
                        cue=True;num_rise+=1
                    gap_history.append((idx,gap_norm))
                    glove_prior=(left['center'],right['center']);last_glove=idx
                    x1,y1=map(int,left['center']);x2,y2=map(int,right['center'])
                    for hand in (left,right):
                        b1,b2,b3,b4=map(int,hand['bbox']);cv2.rectangle(frame,(b1,b2),(b3,b4),CYAN,3)
                    cv2.line(frame,(x1,y1),(x2,y2),CYAN,3)
                    put(frame,f'gap {gap_px:.0f}px / {gap_norm:.3f}',max(5,(x1+x2)//2-115),max(130,(y1+y2)//2-20),CYAN,.72)
                elif elapsed>.32:
                    glove_prior=None;gap_history.clear()
                glove_flags.append(two);rise_flags.append(cue)

                white_flag=status in ('WHITE_REVIEW','MIXED_REVIEW_RISK')
                rows.writerow([idx+1,round(t,3),status,int(direct is not None),int(white_flag),
                               int(status=='SIDE_REVIEW'),int(status in ('PLASTIC_RISK','MIXED_REVIEW_RISK')),
                               round(direct['conf'],4) if direct else '',
                               round(side['conf'],4) if side else '',
                               round(color['area'],4) if color else '',
                               color['anchor_iou'] if color else '',
                               color['solidity'] if color else '',
                               int(two),round(gap_px,1) if two else '',
                               round(gap_norm,4) if two else '',int(cue),
                               round(counts['TOP_DIRECT']/fps,3),
                               round((counts['WHITE_REVIEW']+counts['MIXED_REVIEW_RISK'])/fps,3)])
                text_color={'TOP_DIRECT':GREEN,'WHITE_REVIEW':BLUE,'MIXED_REVIEW_RISK':BLUE,
                            'SIDE_REVIEW':YELLOW,'PLASTIC_RISK':RED,'NO_TOP_EVIDENCE':(195,195,195)}[status]
                cv2.rectangle(frame,(0,0),(w,115),(27,27,27),-1)
                put(frame,f'{t:.2f}s  {status}',15,43,text_color,1.)
                if two:put(frame,f'GLOVES 2 | gap={gap_norm:.3f} | rise={int(cue)} (motion only)',15,82,CYAN,.72)
                else:put(frame,'GLOVES: pair not reliable',15,82,(185,185,185),.75)
                y=h-122;cv2.rectangle(frame,(0,y-7),(w,h),(24,24,24),-1)
                put(frame,'Status history',13,y-11,(240,240,240),.55)
                span=end-start
                for lane,values in enumerate((history,glove_flags,rise_flags)):
                    yy=y+lane*28
                    for j,value in enumerate(values):
                        xx1=int(j*w/span);xx2=max(xx1+1,int((j+1)*w/span))
                        if lane==0:
                            col={'TOP_DIRECT':GREEN,'WHITE_REVIEW':BLUE,'MIXED_REVIEW_RISK':RED,
                                 'SIDE_REVIEW':YELLOW,'PLASTIC_RISK':RED,'NO_TOP_EVIDENCE':GRAY}[value]
                        else:col=(CYAN if lane==1 else PURPLE) if value else (73,73,73)
                        cv2.rectangle(frame,(xx1,yy),(xx2,yy+16),col,-1)
                put(frame,'TOP green | WHITE review cyan | SIDE yellow | plastic red | gloves light cyan',10,h-10,(245,245,245),.61)
                writer.write(frame)
                if processed%30==0:print(f'Progress: {processed}/{end-start}')
    finally:
        cap.release();writer.release()
    print(f'\nProcessed {processed} frames; mutually exclusive display statuses: {counts}')
    print(f'Direct TOP observed: {counts["TOP_DIRECT"]/fps:.2f}s (only direct frames count as model TOP).')
    print(f'White-surface REVIEW: {(counts["WHITE_REVIEW"]+counts["MIXED_REVIEW_RISK"])/fps:.2f}s (NOT verified top).')
    print(f'Glove pairs: {num_pairs}; gap-rise frames: {num_rise}; max normalized gap: {max_gap:.3f}')
    print('A bright white surface is not proof of visual inspection or absence of plastic.')
    print('Video:',video_path,'\nCSV:',csv_path)


if __name__=='__main__':
    main()
