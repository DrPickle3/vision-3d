CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -pthread
SRC = main.cpp
TARGET = Dev2

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

docker-build:
	docker build -t imn269-cpp-$(TARGET) .

docker-run: docker-build
	docker run \
		--rm --privileged \
		-it \
		--network host \
		--pid host \
		imn269-cpp-$(TARGET)

.PHONY: all clean run docker-build docker-run
