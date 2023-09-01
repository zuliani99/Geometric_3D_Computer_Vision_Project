from utils import random_bgr_color

# Polygon class that store useful informations

class Polygon:
	def __init__(self) -> None:
		self.cover = True
		self.circles_ctr_coords = None
		self.vertex_coords = None
		self.point_A = None
		self.extreme_middle_point = None
		self.color = random_bgr_color()

	
	def update_info(self, cover, circles_ctr_coords, vertex_coords, point_A, extreme_middle_point):
		self.cover = cover
		self.circles_ctr_coords = circles_ctr_coords
		self.vertex_coords = vertex_coords
		self.point_A = point_A
		self.extreme_middle_point = extreme_middle_point