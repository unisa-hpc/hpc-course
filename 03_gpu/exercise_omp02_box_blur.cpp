// box_blur_color_openmp_bmp_variable.cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <omp.h>

// the input BMP file must be 24-bit BGR

#pragma pack(push,1)
struct BMPFileHeader {
    uint16_t bfType;      
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};
struct BMPInfoHeader {
    uint32_t biSize;      
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;    
    uint16_t biBitCount;  
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

bool readBMP(const std::string &fname,
                   std::vector<uint8_t> &data,
                   int &W, int &H)
{
    std::ifstream in(fname, std::ios::binary);
    if (!in) return false;
    BMPFileHeader fh; BMPInfoHeader ih;
    in.read(reinterpret_cast<char*>(&fh), sizeof(fh));
    in.read(reinterpret_cast<char*>(&ih), sizeof(ih));
    if (fh.bfType!=0x4D42 || ih.biBitCount!=24) return false;

    W = ih.biWidth;
    H = std::abs(ih.biHeight);
    size_t rowSize = ((W*3 + 3)/4)*4;
    data.resize(rowSize * H);
    for (int r = 0; r < H; ++r) {
        int dst = (ih.biHeight>0) ? (H-1-r) : r;
        in.read(reinterpret_cast<char*>(&data[dst*rowSize]), rowSize);
    }
    return true;
}

bool writeBMP(const std::string &fname,
                    const std::vector<uint8_t> &data,
                    int W, int H)
{
    BMPFileHeader fh={}; BMPInfoHeader ih={};
    size_t rowSize = ((W*3 + 3)/4)*4;
    size_t imgSize = rowSize * H;

    fh.bfType    = 0x4D42;
    fh.bfOffBits = sizeof(fh) + sizeof(ih);
    fh.bfSize    = fh.bfOffBits + imgSize;

    ih.biSize      = sizeof(ih);
    ih.biWidth     = W;
    ih.biHeight    = -H;   // top-down
    ih.biPlanes    = 1;
    ih.biBitCount  = 24;
    ih.biSizeImage = imgSize;

    std::ofstream out(fname, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<char*>(&fh), sizeof(fh));
    out.write(reinterpret_cast<char*>(&ih), sizeof(ih));
    out.write(reinterpret_cast<const char*>(data.data()), imgSize);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.bmp K\n"
                  << "  K = blur kernel size (odd >= 3)\n";
        return 1;
    }

    int W, H;
    std::vector<uint8_t> in, out;
    if (!readBMP(argv[1], in, W, H)) {
        std::cerr << "Error reading " << argv[1] << "\n";
        return 1;
    }

    int K = std::stoi(argv[2]);
    if (K < 3 || (K % 2) == 0) {
        std::cerr << "K must be an odd integer >= 3\n";
        return 1;
    }
    int r = K / 2;                    // radius
    size_t rowSize = ((W*3 + 3)/4)*4;
    out = in;                         // copy borders automatically

    // Box blur KxK: for each pixel, sum KxK pixels
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = r; i < H - r; ++i) {
        for (int j = r; j < W - r; ++j) {
            int sumB = 0, sumG = 0, sumR = 0;
            // sum KxK pixels
            for (int di = -r; di <= r; ++di) {
                for (int dj = -r; dj <= r; ++dj) {
                    size_t idx = (size_t)(i+di)*rowSize + (j+dj)*3;
                    sumB += in[idx + 0];
                    sumG += in[idx + 1];
                    sumR += in[idx + 2];
                }
            }
            size_t dst = (size_t)i*rowSize + j*3;
            out[dst + 0] = static_cast<uint8_t>(sumB / (K*K));
            out[dst + 1] = static_cast<uint8_t>(sumG / (K*K));
            out[dst + 2] = static_cast<uint8_t>(sumR / (K*K));
        }
    }

    if (!writeBMP("out.bmp", out, W, H)) {
        std::cerr << "Error writing out.bmp\n";
        return 1;
    }

    std::cout << "Applied box blur " << K << "x" << K 
              << ", output: out.bmp (" << W << "x" << H << ")\n";
    return 0;
}