from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

class PlayerReidentifier:
    def __init__(self, model_path='best.pt', video_path='15sec_input_720p.mp4'):
        self.model = YOLO(model_path)
        
        self.tracker = DeepSort(
            max_age=50,
            n_init=3,
            max_iou_distance=0.3,
            nn_budget=100,
            embedder="mobilenet", 
            half=True, 
            bgr=True
        )
        
        self.cap = cv2.VideoCapture(video_path)
        
        self.CONFIDENCE_THRESHOLD = 0.6
        self.NMS_THRESHOLD = 0.4
        self.SKIP_FRAMES = 1
        
        self.id_mapping = {}
        self.next_available_id = 1
        self.id_history = defaultdict(list)
        self.lost_players = {}
        
        self.frame_count = 0
        self.detection_count = 0
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def preprocess_detections(self, results):
        """Enhanced detection preprocessing with NMS and filtering"""
        detections = []
        boxes_data = results.boxes.data.tolist()
        
        if not boxes_data:
            return detections
            
        boxes = []
        confidences = []
        valid_detections = []
        
        for box in boxes_data:
            x1, y1, x2, y2, conf, cls = box
            
            if int(cls) in [1, 2] and conf >= self.CONFIDENCE_THRESHOLD:
                width, height = x2 - x1, y2 - y1
                if width > 15 and height > 30:
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
                    valid_detections.append(box)
        
        if not boxes:
            return detections
            
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                  self.CONFIDENCE_THRESHOLD, 
                                  self.NMS_THRESHOLD)
        
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                x1, y1, x2, y2, conf, cls = valid_detections[i]
                player_type = "goalkeeper" if int(cls) == 1 else "player"
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, player_type))
                
        return detections
    
    def manage_player_ids(self, tracks):
        """Enhanced ID management with re-identification logic"""
        current_frame_ids = set()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            original_id = track.track_id
            current_frame_ids.add(original_id)
            
            ltrb = track.to_ltrb()
            center_x, center_y = (ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2
            
            if original_id not in self.id_mapping:
                best_match_id = self.find_returning_player(center_x, center_y, ltrb)
                
                if best_match_id is not None:
                    self.id_mapping[original_id] = best_match_id
                    print(f"Re-identified returning player: Track {original_id} -> Player {best_match_id}")
                else:
                    self.id_mapping[original_id] = self.next_available_id
                    print(f"New player detected: Track {original_id} -> Player {self.next_available_id}")
                    self.next_available_id += 1
            
            local_id = self.id_mapping[original_id]
            self.id_history[local_id].append((center_x, center_y, self.frame_count))
            
            if len(self.id_history[local_id]) > 30:
                self.id_history[local_id] = self.id_history[local_id][-30:]
        
        self.handle_lost_tracks(current_frame_ids)
    
    def find_returning_player(self, center_x, center_y, bbox, max_distance=100):
        """Find if current detection matches a previously lost player"""
        if not self.lost_players:
            return None
            
        best_match = None
        min_distance = float('inf')
        
        for lost_id, (last_pos, last_frame) in self.lost_players.items():
            if self.frame_count - last_frame > 30:
                continue
                
            last_x, last_y = last_pos
            distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
            
            if distance < max_distance and distance < min_distance:
                min_distance = distance
                best_match = lost_id
        
        if best_match is not None:
            del self.lost_players[best_match]
            
        return best_match
    
    def handle_lost_tracks(self, current_frame_ids):
        """Handle tracks that are no longer active"""
        all_tracked_ids = set(self.id_mapping.keys())
        lost_track_ids = all_tracked_ids - current_frame_ids
        
        for lost_track_id in lost_track_ids:
            if lost_track_id in self.id_mapping:
                local_id = self.id_mapping[lost_track_id]
                
                if local_id in self.id_history and self.id_history[local_id]:
                    last_pos = self.id_history[local_id][-1][:2]
                    self.lost_players[local_id] = (last_pos, self.frame_count)
                
                del self.id_mapping[lost_track_id]
    
    def draw_enhanced_annotations(self, frame, tracks):
        """Draw enhanced bounding boxes and information"""
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            original_id = track.track_id
            if original_id not in self.id_mapping:
                continue
                
            local_id = self.id_mapping[original_id]
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
            color = colors[local_id % len(colors)]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"Player {local_id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), color, -1)
            
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if local_id in self.id_history and len(self.id_history[local_id]) > 1:
                points = self.id_history[local_id][-10:]
                for i in range(1, len(points)):
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    cv2.line(frame, pt1, pt2, color, 1)
    
    def add_info_overlay(self, frame):
        """Add information overlay to the frame"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        info_text = [
            f"Frame: {self.frame_count}",
            f"Active Players: {len(self.id_mapping)}",
            f"Total Detected: {max(self.id_mapping.values()) if self.id_mapping else 0}",
            f"Lost Players: {len(self.lost_players)}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (15, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_video(self, output_path=None, show_display=True):
        """Main processing loop with optional video output"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        print("Starting player re-identification...")
        print(f"Video: {self.width}x{self.height} @ {self.fps}fps")
        
        start_time = time.time()
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if self.frame_count % self.SKIP_FRAMES == 0:
                    results = self.model.predict(frame, verbose=False, conf=0.3)[0]
                    
                    detections = self.preprocess_detections(results)
                    self.detection_count += len(detections)
                    
                    tracks = self.tracker.update_tracks(detections, frame=frame)
                    
                    self.manage_player_ids(tracks)
                else:
                    tracks = self.tracker.tracks
                
                self.draw_enhanced_annotations(frame, tracks)
                self.add_info_overlay(frame)
                
                if out is not None:
                    out.write(frame)
                
                if show_display:
                    cv2.imshow('Enhanced Player Re-Identification', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.frame_count += 1
                
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = self.frame_count / elapsed
                    active_players = len([t for t in tracks if t.is_confirmed()])
                    print(f"Processed {self.frame_count} frames | FPS: {fps:.1f} | Active tracks: {active_players}")
        
        finally:
            self.cleanup(out)
            
        self.print_statistics(time.time() - start_time)
    
    def cleanup(self, out=None):
        """Clean up resources"""
        self.cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
    
    def print_statistics(self, total_time):
        """Print processing statistics"""
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {self.detection_count}")
        print(f"Unique players identified: {max(self.id_mapping.values()) if self.id_mapping else 0}")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"Average FPS: {self.frame_count / total_time:.2f}")
        print(f"ID mapping: {dict(self.id_mapping)}")

if __name__ == "__main__":
    reid_system = PlayerReidentifier(
        model_path='best.pt',
        video_path='15sec_input_720p.mp4'
    )
    
    reid_system.process_video(
        output_path=None,
        show_display=True
    )