import numpy as np
import matplotlib.pyplot as plt
import os

class CameraPositioning:
    def __init__(self, field_length=100, field_width=60):
        """
        Initialize with field dimensions in yards
        Standard football pitch: 100-130 yards × 50-100 yards
        """
        self.field_length = field_length  # yards
        self.field_width = field_width   # yards

    def calculate_optimal_positions(self):
        """
        Calculate optimal camera positions for different setups
        """
        positions = {
            'main_tactical': {
                'x': self.field_length / 2,  # Center line
                'y': -25,  # 25 yards back from sideline
                'z': 12,   # 12 feet high
                'description': 'Primary tactical analysis camera'
            },
            'goal_a': {
                'x': -15,  # 15 yards behind goal
                'y': self.field_width / 2,
                'z': 10,
                'description': 'Goal A detection camera'
            },
            'goal_b': {
                'x': self.field_length + 15,
                'y': self.field_width / 2,
                'z': 10,
                'description': 'Goal B detection camera'
            },
            'corner_overview': {
                'x': self.field_length + 20,
                'y': -20,
                'z': 15,
                'description': 'Corner overview camera'
            }
        }
        return positions

    def calculate_coverage_area(self, camera_pos, fov_horizontal=120, fov_vertical=90):
        """
        Calculate the field area covered by camera
        FOV in degrees (GoPro wide angle ~120° horizontal)
        """
        x, y, z = camera_pos['x'], camera_pos['y'], camera_pos['z']

        # Convert FOV to radians
        fov_h_rad = np.radians(fov_horizontal / 2)
        fov_v_rad = np.radians(fov_vertical / 2)

        # Calculate coverage at ground level (z=0)
        coverage_distance = abs(z) / np.tan(fov_v_rad)
        coverage_width = 2 * coverage_distance * np.tan(fov_h_rad)

        return {
            'distance_covered': coverage_distance,
            'width_covered': coverage_width,
            'area_covered': coverage_distance * coverage_width
        }

    def optimize_height_for_coverage(self, desired_coverage_distance, fov_vertical=90):
        """
        Calculate optimal camera height for desired field coverage
        """
        fov_v_rad = np.radians(fov_vertical / 2)
        optimal_height = desired_coverage_distance * np.tan(fov_v_rad)
        return optimal_height

    def plot_field_layout(self, camera_positions):
        """
        Visualize field and camera positions
        """
        plt.figure(figsize=(12, 8))

        # Draw field
        field_rect = plt.Rectangle((0, 0), self.field_length, self.field_width,
                                 fill=False, edgecolor='green', linewidth=2, label='Football Field')
        plt.gca().add_patch(field_rect)

        # Draw center line
        plt.axvline(x=self.field_length/2, color='green', linestyle='--', alpha=0.7, label='Center Line')

        # Draw goals
        goal_width = 8  # yards
        goal_margin = (self.field_width - goal_width) / 2

        goal_a = plt.Rectangle((0, goal_margin), -2, goal_width,
                              fill=False, edgecolor='red', linewidth=2)
        goal_b = plt.Rectangle((self.field_length, goal_margin), 2, goal_width,
                              fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(goal_a)
        plt.gca().add_patch(goal_b)

        # Plot camera positions
        colors = ['blue', 'orange', 'purple', 'brown']
        for i, (name, pos) in enumerate(camera_positions.items()):
            plt.plot(pos['x'], pos['y'], 'o', color=colors[i % len(colors)],
                   markersize=12, label=f"{name.replace('_', ' ').title()} (H: {pos['z']}ft)")

            # Draw coverage area
            coverage = self.calculate_coverage_area(pos)
            coverage_radius = min(coverage['distance_covered'], self.field_length) / 2

            circle = plt.Circle((pos['x'], pos['y']), coverage_radius,
                              fill=False, color=colors[i % len(colors)],
                              alpha=0.3, linestyle='--')
            plt.gca().add_patch(circle)

            # Add text label
            plt.annotate(f"{name.replace('_', ' ').title()}",
                        (pos['x'], pos['y']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, weight='bold')

        plt.xlim(-30, self.field_length + 30)
        plt.ylim(-30, self.field_width + 30)
        plt.xlabel('Field Length (yards)', fontsize=12)
        plt.ylabel('Field Width (yards)', fontsize=12)
        plt.title('Optimal Camera Positions for Football Analytics', fontsize=14, weight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')

        # Save the plot
        plt.tight_layout()
        plt.savefig('camera_positions_layout.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n📸 Camera layout diagram saved as 'camera_positions_layout.png'")
        return plt.gcf()

def main():
    """
    Main function to run camera positioning analysis
    """
    print("🏈 FOOTBALL ANALYTICS - CAMERA POSITIONING TOOL")
    print("=" * 50)

    # Initialize positioning calculator
    positioning = CameraPositioning()
    positions = positioning.calculate_optimal_positions()

    print("\n📋 OPTIMAL CAMERA POSITIONS:")
    print("-" * 30)

    for name, pos in positions.items():
        print(f"\n🎥 {name.upper().replace('_', ' ')}:")
        print(f"   Position: ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f})")
        print(f"   Description: {pos['description']}")

        coverage = positioning.calculate_coverage_area(pos)
        print(f"   Coverage: {coverage['distance_covered']:.1f} yards distance")
        print(f"   Width: {coverage['width_covered']:.1f} yards")
        print(f"   Total Area: {coverage['area_covered']:.0f} sq yards")

    # Create and save visualization
    positioning.plot_field_layout(positions)

    print("\n✅ SETUP COMPLETE!")
    print("Next steps:")
    print("1. Use the generated diagram to position your cameras")
    print("2. Adjust positions based on your actual field dimensions")
    print("3. Record test footage from each position")

    return positions

if __name__ == "__main__":
    main()
