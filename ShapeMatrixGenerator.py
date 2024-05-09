import numpy as np
from PIL import Image, ImageDraw


class ShapeMatrixGenerator:
    def __init__(self, size=5):
        """
        Initializes the ShapeMatrixGenerator with a specified matrix size.
        """
        self.size = size

    def draw_circle(self):
        """
        Draws a circle in the matrix touching the edges.
        """
        image = Image.new("L", (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((0, 0, self.size - 1, self.size - 1), fill="black")
        matrix = np.array(image) < 128
        return matrix.astype(int)

    def draw_square(self):
        """
        Draws a square filling the matrix.
        """
        image = Image.new("L", (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.size - 1, self.size - 1), fill="black")
        matrix = np.array(image) < 128
        return matrix.astype(int)

    def draw_triangle(self):
        """
        Draws an upward-pointing triangle filling the matrix.
        """
        image = Image.new("L", (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.polygon(
            [(self.size // 2, 0), (0, self.size - 1), (self.size - 1, self.size - 1)],
            fill="black",
        )
        matrix = np.array(image) < 128
        return matrix.astype(int)

    def draw_diamond(self):
        """
        Draws a diamond shape in the matrix.
        """
        image = Image.new("L", (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.polygon(
            [
                (self.size // 2, 0),
                (0, self.size // 2),
                (self.size // 2, self.size - 1),
                (self.size - 1, self.size // 2),
            ],
            fill="black",
        )
        matrix = np.array(image) < 128
        return matrix.astype(int)

    def rotate_matrix(self, matrix, degrees):
        """
        Rotates the matrix by the specified degrees: 0, 90, 180, or 270.
        """
        if degrees == 90:
            return np.rot90(matrix, k=1)
        elif degrees == 180:
            return np.rot90(matrix, k=2)
        elif degrees == 270:
            return np.rot90(matrix, k=3)
        else:
            return matrix


# Example usage
generator = ShapeMatrixGenerator(size=5)
circle_matrix = generator.draw_circle()
square_matrix = generator.draw_square()
triangle_matrix = generator.draw_triangle()
diamond_matrix = generator.draw_diamond()

# To display the matrices, you can print them like this:
if __name__ == "__main__":
    print("Circle Matrix:")
    print(circle_matrix)
    print("Square Matrix:")
    print(square_matrix)
    print("Triangle Matrix:")
    print(triangle_matrix)
    print("Diamond Matrix:")
    print(diamond_matrix)
