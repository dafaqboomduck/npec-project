import numpy as np

class PIDController:
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0) -> None:
        """
        PID Controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        """Reset the controller state."""
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float, output_limits: tuple = None) -> float:
        """
        Compute the PID control output.
        
        Args:
            error: Current error (setpoint - measured value)
            dt: Time step for integration/differentiation
            
        Returns:
            Control output
        """
        # Proportional term
        p = self.kp * error

        
        # Derivative term
        d = self.kd * (error - self.prev_error) / dt

        self.prev_error = error

        # Integral term (with anti-windup)
        i = self.ki * self.integral

        # Calculate output
        output = p + i + d
        
        # Anti-windup: only integrate if we're not saturated
        if output_limits is not None:
            output_clamped = np.clip(output, output_limits[0], output_limits[1])
            
            # Only integrate if output is not saturated, OR if error would reduce saturation
            if output == output_clamped:
                # Not saturated, integrate normally
                self.integral += error * dt
            elif (output > output_limits[1] and error < 0) or (output < output_limits[0] and error > 0):
                # Saturated, but error is pushing back toward valid range
                self.integral += error * dt
            # else: saturated and error is pushing further into saturation - don't integrate
            
            return output_clamped
        else:
            self.integral += error * dt
            return output