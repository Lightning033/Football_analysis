
"""
football_video_analyzer.py
Football Analytics System for Video File Analysis
Fixed version with proper indentation and error handling
"""

import os
import json
from datetime import datetime
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import pytesseract
from tqdm import tqdm
from ultralytics import YOLO


class FootballVideoAnalyzer:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize Football Video Analyzer

        Parameters:
        -----------
        model_path : str
            Path to YOLOv8 model file (yolov8n.pt for speed, yolov8m/l/x.pt for accuracy)
        """
        print(f"🔄 Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)

        # Analysis settings
        self.confidence_threshold = 0.5
        self.track_history_length = 30
        self.next_track_id = 1

        # Tracking data
        self.player_positions = {}
        self.ball_history = []

        print("✅ Football Video Analyzer initialized successfully!")

    def analyze_full_video(self, video_path: str, output_dir: str = "analysis_results") -> Dict:
        """
        Analyze complete video file and generate comprehensive reports

        Parameters:
        -----------
        video_path : str
            Path to the video file to analyze
        output_dir : str
            Directory to save analysis results

        Returns:
        --------
        Dict
            Complete analysis report
        """
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ Video file not found: {video_path}")

        print(f"🏈 Starting analysis of: {os.path.basename(video_path)}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Cannot open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_minutes = total_frames / (fps * 60)

        print(f"📊 Video Info:")
        print(f"   • Duration: {duration_minutes:.1f} minutes")
        print(f"   • Total Frames: {total_frames:,}")
        print(f"   • FPS: {fps:.1f}")
        print(f"   • Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        # Initialize statistics
        match_stats = {
            'total_passes': 0,
            'total_goals': 0,
            'total_saves': 0,
            'total_juggling': 0,
            'player_detections': {},
            'events_timeline': [],
            'video_info': {
                'duration_minutes': duration_minutes,
                'total_frames': total_frames,
                'fps': fps,
                'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            }
        }

        # Process video with progress bar
        frame_count = 0
        progress_bar = tqdm(total=total_frames, desc="🎬 Processing video", unit="frame")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_count / fps
                frame_analysis = self._process_frame(frame, timestamp, frame_count)
                self._update_match_stats(match_stats, frame_analysis)

                frame_count += 1
                progress_bar.update(1)

                # Optional: Skip frames for faster processing
                # Uncomment the next line to process every 3rd frame (3x faster)
                # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + 2)

        except Exception as e:
            print(f"⚠️ Error during processing at frame {frame_count}: {e}")
        finally:
            cap.release()
            progress_bar.close()

        print(f"✅ Processed {frame_count:,} frames successfully!")

        # Generate final report
        final_report = self._generate_final_report(match_stats)

        # Save results
        self._save_analysis_results(final_report, output_dir, video_path)

        return final_report

    def _process_frame(self, frame: np.ndarray, timestamp: float, frame_number: int) -> Dict:
        """Process single frame and extract analytics"""
        # Detect objects
        detections = self._detect_objects(frame)

        # Track objects
        tracked_detections = self._simple_tracking(detections, timestamp)

        # Initialize frame data
        frame_data = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'players': {},
            'ball_position': None,
            'events': []
        }

        # Process detections
        for detection in tracked_detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            track_id = detection.get('track_id')

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            if class_name == 'ball':
                frame_data['ball_position'] = (center_x, center_y)
                self._update_ball_history((center_x, center_y), timestamp)

            elif class_name == 'person' and track_id:
                jersey_number = self._detect_jersey_numbers(frame, bbox)
                frame_data['players'][track_id] = {
                    'position': (center_x, center_y),
                    'bbox': bbox,
                    'jersey_number': jersey_number,
                    'timestamp': timestamp
                }

        # Detect events
        frame_data['events'] = self._detect_all_events(frame_data)

        return frame_data

    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect players and ball using YOLOv8"""
        results = self.model(frame, verbose=False)[0]
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())

                # Filter for relevant classes and confidence
                if confidence > self.confidence_threshold:
                    if class_id == 0:  # person
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_name': 'person'
                        })
                    elif class_id == 32:  # sports ball
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_name': 'ball'
                        })

        return detections

    def _simple_tracking(self, detections: List[Dict], timestamp: float) -> List[Dict]:
        """Simple tracking based on position similarity"""
        tracked_detections = []

        for detection in detections:
            if detection['class_name'] == 'person':
                bbox = detection['bbox']
                current_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]

                # Find best match with previous positions
                best_match_id = None
                min_distance = float('inf')

                for track_id, prev_data in self.player_positions.items():
                    if timestamp - prev_data['last_seen'] > 2.0:  # Remove old tracks
                        continue

                    distance = np.sqrt(
                        (current_center[0] - prev_data['position'][0])**2 +
                        (current_center[1] - prev_data['position'][1])**2
                    )

                    if distance < min_distance and distance < 150:  # pixel threshold
                        min_distance = distance
                        best_match_id = track_id

                # Assign track ID
                if best_match_id is None:
                    track_id = self.next_track_id
                    self.next_track_id += 1
                else:
                    track_id = best_match_id

                # Update tracking
                self.player_positions[track_id] = {
                    'position': current_center,
                    'last_seen': timestamp
                }

                detection['track_id'] = track_id

            tracked_detections.append(detection)

        return tracked_detections

    def _detect_jersey_numbers(self, frame: np.ndarray, bbox: List[float]) -> str:
        """Extract jersey number using OCR"""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]

            # Crop player region
            player_crop = frame[y1:y2, x1:x2]

            if player_crop.size == 0:
                return ""

            # Focus on upper torso for jersey number
            height, width = player_crop.shape[:2]
            torso_crop = player_crop[
                int(height*0.1):int(height*0.5),
                int(width*0.3):int(width*0.7)
            ]

            if torso_crop.size == 0:
                return ""

            # Convert to grayscale and enhance
            gray = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray)

            # OCR configuration for numbers only
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(enhanced, config=custom_config)

            # Clean and validate result
            number = ''.join(filter(str.isdigit, text.strip()))
            return number if 0 < len(number) <= 2 else ""

        except Exception:
            return ""

    def _update_ball_history(self, position, timestamp):
        """Update ball position history"""
        self.ball_history.append({
            'position': position,
            'timestamp': timestamp
        })

        # Keep only recent history
        if len(self.ball_history) > self.track_history_length:
            self.ball_history.pop(0)

    def _detect_all_events(self, frame_data: Dict) -> List[str]:
        """Detect all types of events in current frame"""
        events = []

        if self._detect_pass(frame_data):
            events.append('PASS')

        if self._detect_goal(frame_data):
            events.append('GOAL')

        if self._detect_save(frame_data):
            events.append('SAVE')

        if self._detect_juggling():
            events.append('JUGGLING')

        return events

    def _detect_pass(self, frame_data: Dict) -> bool:
        """Detect pass events based on ball movement"""
        if len(self.ball_history) < 3 or not frame_data['ball_position']:
            return False

        # Analyze recent ball movement
        recent_positions = [entry['position'] for entry in self.ball_history[-3:]]

        # Check if ball moved significantly
        movement_distance = np.linalg.norm(
            np.array(recent_positions[-1]) - np.array(recent_positions[0])
        )

        # Check if players are nearby
        nearby_players = 0
        ball_pos = frame_data['ball_position']

        for player_data in frame_data['players'].values():
            player_pos = player_data['position']
            distance = np.linalg.norm(np.array(ball_pos) - np.array(player_pos))
            if distance < 100:  # pixels
                nearby_players += 1

        return movement_distance > 50 and nearby_players > 0

    def _detect_goal(self, frame_data: Dict) -> bool:
        """Detect goal events"""
        if not frame_data['ball_position']:
            return False

        ball_x, ball_y = frame_data['ball_position']

        # Goal area detection (adjust coordinates based on your camera setup)
        goal_areas = [
            {'x_range': (0, 150), 'y_range': (250, 450)},      # Left goal area
            {'x_range': (1770, 1920), 'y_range': (250, 450)}   # Right goal area
        ]

        for goal_area in goal_areas:
            if (goal_area['x_range'][0] <= ball_x <= goal_area['x_range'][1] and
                goal_area['y_range'][0] <= ball_y <= goal_area['y_range'][1]):
                return True

        return False

    def _detect_save(self, frame_data: Dict) -> bool:
        """Detect goalkeeper save attempts (simplified implementation)"""
        # This is a simplified placeholder - would need more sophisticated logic
        # to identify goalkeepers and detect save actions
        return False

    def _detect_juggling(self) -> bool:
        """Detect ball juggling based on vertical oscillations"""
        if len(self.ball_history) < 5:
            return False

        # Get recent vertical positions
        y_positions = [entry['position'][1] for entry in self.ball_history[-5:]]

        # Count direction changes in vertical movement
        direction_changes = 0
        for i in range(len(y_positions) - 2):
            curr_diff = y_positions[i+1] - y_positions[i]
            next_diff = y_positions[i+2] - y_positions[i+1]

            # Sign change indicates direction change
            if curr_diff * next_diff < 0:
                direction_changes += 1

        return direction_changes >= 2

    def _update_match_stats(self, match_stats: Dict, frame_analysis: Dict):
        """Update overall match statistics with frame analysis"""
        # Count events
        for event in frame_analysis['events']:
            if event == 'PASS':
                match_stats['total_passes'] += 1
            elif event == 'GOAL':
                match_stats['total_goals'] += 1
            elif event == 'SAVE':
                match_stats['total_saves'] += 1
            elif event == 'JUGGLING':
                match_stats['total_juggling'] += 1

            # Add to timeline
            match_stats['events_timeline'].append({
                'event': event,
                'timestamp': frame_analysis['timestamp'],
                'minute': int(frame_analysis['timestamp'] / 60),
                'frame': frame_analysis['frame_number']
            })

        # Track player detections
        for track_id, player_data in frame_analysis['players'].items():
            if track_id not in match_stats['player_detections']:
                match_stats['player_detections'][track_id] = {
                    'total_detections': 0,
                    'jersey_numbers': set(),
                    'first_seen': frame_analysis['timestamp'],
                    'last_seen': frame_analysis['timestamp']
                }

            player_stats = match_stats['player_detections'][track_id]
            player_stats['total_detections'] += 1
            player_stats['last_seen'] = frame_analysis['timestamp']

            if player_data['jersey_number']:
                player_stats['jersey_numbers'].add(player_data['jersey_number'])

    def _generate_final_report(self, match_stats: Dict) -> Dict:
        """Generate comprehensive final analysis report"""
        # Convert sets to lists for JSON serialization
        for player_id, player_data in match_stats['player_detections'].items():
            player_data['jersey_numbers'] = list(player_data['jersey_numbers'])

        # Calculate additional statistics
        total_events = (
            match_stats['total_passes'] +
            match_stats['total_goals'] +
            match_stats['total_saves'] +
            match_stats['total_juggling']
        )

        duration_minutes = match_stats['video_info']['duration_minutes']

        final_report = {
            'analysis_summary': {
                'total_events': total_events,
                'events_per_minute': round(total_events / duration_minutes, 2) if duration_minutes > 0 else 0,
                'unique_players_detected': len(match_stats['player_detections']),
                'analysis_date': datetime.now().isoformat(),
                'processing_info': {
                    'model_used': 'YOLOv8',
                    'confidence_threshold': self.confidence_threshold
                }
            },
            'event_breakdown': {
                'passes': match_stats['total_passes'],
                'goals': match_stats['total_goals'],
                'saves': match_stats['total_saves'],
                'juggling_sequences': match_stats['total_juggling']
            },
            'timeline': match_stats['events_timeline'],
            'players': match_stats['player_detections'],
            'video_info': match_stats['video_info']
        }

        return final_report

    def _save_analysis_results(self, report: Dict, output_dir: str, video_path: str):
        """Save analysis results to files"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_filename = f"{video_name}_analysis_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save CSV summary
        csv_data = {
            'Event_Type': ['Passes', 'Goals', 'Saves', 'Juggling'],
            'Count': [
                report['event_breakdown']['passes'],
                report['event_breakdown']['goals'],
                report['event_breakdown']['saves'],
                report['event_breakdown']['juggling_sequences']
            ]
        }

        csv_filename = f"{video_name}_summary_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        # Save timeline CSV
        if report['timeline']:
            timeline_df = pd.DataFrame(report['timeline'])
            timeline_filename = f"{video_name}_timeline_{timestamp}.csv"
            timeline_path = os.path.join(output_dir, timeline_filename)
            timeline_df.to_csv(timeline_path, index=False)

        # Print save locations
        print(f"\n📊 Analysis Results Saved:")
        print(f"📄 Detailed Report: {json_path}")
        print(f"📊 Summary CSV: {csv_path}")
        if report['timeline']:
            print(f"⏱️ Timeline CSV: {timeline_path}")

        # Print summary to console
        self._print_console_summary(report)

    def _print_console_summary(self, report: Dict):
        """Print analysis summary to console"""
        print(f"\n🏈 FOOTBALL MATCH ANALYSIS COMPLETE")
        print("=" * 50)

        summary = report['analysis_summary']
        events = report['event_breakdown']

        print(f"⏱️  Video Duration: {report['video_info']['duration_minutes']:.1f} minutes")
        print(f"🎬 Total Frames: {report['video_info']['total_frames']:,}")
        print(f"👥 Players Detected: {summary['unique_players_detected']}")
        print(f"🎯 Total Events: {summary['total_events']}")
        print(f"📈 Events/Minute: {summary['events_per_minute']}")

        print(f"\n📊 EVENT BREAKDOWN:")
        print(f"⚽ Passes: {events['passes']}")
        print(f"🥅 Goals: {events['goals']}")
        print(f"🥊 Saves: {events['saves']}")
        print(f"🤹 Juggling: {events['juggling_sequences']}")

        # Show most active players if available
        if report['players']:
            print(f"\n👥 TOP 3 MOST DETECTED PLAYERS:")
            sorted_players = sorted(
                report['players'].items(),
                key=lambda x: x[1]['total_detections'],
                reverse=True
            )[:3]

            for i, (player_id, data) in enumerate(sorted_players, 1):
                jersey = data['jersey_numbers'][0] if data['jersey_numbers'] else 'Unknown'
                detections = data['total_detections']
                print(f"  {i}. Player #{jersey} (ID:{player_id}) - {detections:,} detections")


def analyze_match_video(video_path: str, output_dir: str = "match_analysis") -> Dict:
    """
    Convenience function to analyze a match video

    Parameters:
    -----------
    video_path : str
        Path to the video file
    output_dir : str
        Output directory for results

    Returns:
    --------
    Dict
        Analysis results
    """
    analyzer = FootballVideoAnalyzer()
    return analyzer.analyze_full_video(video_path, output_dir)


# Main execution
if __name__ == "__main__":
    # Video file path - UPDATE THIS WITH YOUR VIDEO PATH
    video_path = r"C:\Users\Harsh kansara\PycharmProjects\Clubduelz\Uploads\normal 3.mp4"

    # Output directory
    output_directory = r"C:\Users\Harsh kansara\PycharmProjects\CZ\analysis_results"

    # Check if video file exists
    if os.path.exists(video_path):
        print(f"✅ Found video file: {os.path.basename(video_path)}")
        print(f"📁 Output will be saved to: {output_directory}")

        try:
            # Run the analysis
            results = analyze_match_video(video_path, output_directory)
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📁 Check the '{output_directory}' folder for detailed results.")

        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            print("Please check your video file and try again.")

    else:
        print(f"❌ Video file not found: {video_path}")
        print("Please check the file path and make sure the video exists.")
        print("\nSupported formats: MP4, AVI, MOV, MKV")
