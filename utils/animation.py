"""
Animation saving utility for matplotlib-based simulations.
"""

import os
import glob
import subprocess
import matplotlib.pyplot as plt


class AnimationSaver:
    """Handles saving matplotlib animations as video files."""
    
    def __init__(self, output_dir="output/animations", save_per_frame=1, fps=30):
        """
        Initialize the animation saver.
        
        Args:
            output_dir: Directory to save animation frames and video
            save_per_frame: Save every N frames (1 = save all)
            fps: Frames per second for output video
        """
        self.output_dir = output_dir
        self.save_per_frame = save_per_frame
        self.fps = fps
        self.frame_idx = 0
        self.enabled = True
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_frame(self, fig, force=False):
        """
        Save current frame if conditions are met.
        
        Args:
            fig: Matplotlib figure to save
            force: Force save regardless of frame count
        """
        if not self.enabled:
            return
            
        self.frame_idx += 1
        if force or self.frame_idx % self.save_per_frame == 0:
            frame_num = self.frame_idx // self.save_per_frame
            filepath = os.path.join(self.output_dir, f"frame_{frame_num:05d}.png")
            fig.savefig(filepath, dpi=150)
    
    def export_video(self, output_name="animation.mp4", cleanup=True):
        """
        Export saved frames as video using ffmpeg.
        
        Args:
            output_name: Name of output video file
            cleanup: Whether to delete PNG files after export
        """
        if not self.enabled:
            return
            
        input_pattern = os.path.join(self.output_dir, "frame_%05d.png")
        output_path = os.path.join(self.output_dir, output_name)
        
        try:
            subprocess.call([
                'ffmpeg', '-y',
                '-framerate', str(self.fps),
                '-i', input_pattern,
                '-vf', f'scale=-2:720,fps={self.fps}',
                '-pix_fmt', 'yuv420p',
                '-c:v', 'libx264',
                output_path
            ])
            print(f"Video saved to: {output_path}")
        except Exception as e:
            print(f"Failed to export video: {e}")
            return
        
        if cleanup:
            for f in glob.glob(os.path.join(self.output_dir, "frame_*.png")):
                os.remove(f)
    
    def reset(self):
        """Reset frame counter."""
        self.frame_idx = 0

