if self.is_obstacle_in_relative_direction(1):
    self.left()
elif self.is_obstacle_in_relative_direction(0):
    self.right()
elif self.is_obstacle_in_relative_direction(-1):
    self.right()
else:
    self.ahead()
