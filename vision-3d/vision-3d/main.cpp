#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <iostream>

using namespace std;

void createImageWithBlackSquares(const char* filename, int width, int height, int squareSize, int spacing) {
    int channels = 3; // RGB
    std::vector<unsigned char> image(width * height * channels, 255); // fond blanc

    // Dessin des carrés noirs
    for (int y = 0; y < height; y += squareSize + spacing) {
        for (int x = 0; x < width; x += squareSize + spacing) {
            for (int dy = 0; dy < squareSize && (y + dy) < height; ++dy) {
                for (int dx = 0; dx < squareSize && (x + dx) < width; ++dx) {
                    int pixelIndex = ((y + dy) * width + (x + dx)) * channels;
                    image[pixelIndex + 0] = 0; // R
                    image[pixelIndex + 1] = 0; // G
                    image[pixelIndex + 2] = 0; // B
                }
            }
        }
    }

    // Écriture de l'image
    if (!stbi_write_png(filename, width, height, channels, image.data(), width * channels)) {
        std::cerr << "Erreur lors de l'écriture de l'image PNG !" << std::endl;
    } else {
        std::cout << "Image sauvegardée dans : " << filename << std::endl;
    }
}

int main() {
	createImageWithBlackSquares("output.png", 2480, 3508, 100, 100);
	return 0;
}