CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wno-deprecated-enum-enum-conversion -O2 -pthread `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
SRC = main.cpp
TARGET = LumiereStructuree

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
