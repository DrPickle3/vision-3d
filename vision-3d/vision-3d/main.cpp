#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;

void divideFileByTen(const std::string& inputFilename, const std::string& outputFilename) {
    std::ifstream inputFile(inputFilename);
    std::ofstream outputFile(outputFilename);

    if (!inputFile.is_open()) {
        std::cerr << "Error: Cannot open input file.\n";
        return;
    }
    if (!outputFile.is_open()) {
        std::cerr << "Error: Cannot open output file.\n";
        return;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> dividedValues;

        while (std::getline(ss, value, ',')) {
            try {
                double num = std::stod(value);
                num *= 10.0;

                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << num;
                dividedValues.push_back(oss.str());
            } catch (...) {
                dividedValues.push_back("NaN"); // Optional: handle malformed data
            }
        }

        for (size_t i = 0; i < dividedValues.size(); ++i) {
            outputFile << dividedValues[i];
            if (i < dividedValues.size() - 1) {
                outputFile << ",";
            }
        }
        outputFile << "\n";
    }

    inputFile.close();
    outputFile.close();
    std::cout << "File processed and saved to '" << outputFilename << "'.\n";
}

int main() {
	cout << "Hello World !";
	divideFileByTen("points.txt", "Truepoints.txt");
	return 0;
}