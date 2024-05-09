import numpy as np
from PIL import Image, ImageDraw, ImageFont


class TextMatrixGenerator:
    def __init__(self, size=10):
        """
        Initializes the TextMatrixGenerator with a specified matrix size.
        """
        self.size = size

    def text_to_matrix(self, text):
        """
        Converts a text character to a binary matrix of a predefined size.
        """
        image = Image.new("L", (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        # Calculate the bounding box for the text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Center the text
        text_top_left = ((self.size - text_width) // 2, (self.size - text_height) // 2)

        # Render the text in black
        draw.text(text_top_left, text, fill="black", font=font)

        # Convert to numpy array and threshold to create a binary matrix
        matrix = np.array(image) < 128

        return matrix.astype(int)

    def scale_up_matrix(self, small_matrix, scale_factor):
        """
        Scales up a binary matrix by a given factor.
        """
        large_size = small_matrix.shape[0] * scale_factor
        large_matrix = np.zeros((large_size, large_size), dtype=int)

        for i in range(large_size):
            for j in range(large_size):
                small_i, small_j = i // scale_factor, j // scale_factor
                large_matrix[i, j] = small_matrix[small_i, small_j]

        return large_matrix

    def rotate_matrix(self, matrix, degrees):
        if degrees == 90:
            return np.rot90(matrix, k=1)
        elif degrees == 180:
            return np.rot90(matrix, k=2)
        elif degrees == 270:
            return np.rot90(matrix, k=3)
        else:
            return matrix


# # Example usage
# generator = TextMatrixGenerator(size=10)
# small_matrix = generator.text_to_matrix('A')
# print(small_matrix)
# scaled_matrix = generator.scale_up_matrix(small_matrix, 2)
# print(scaled_matrix)
