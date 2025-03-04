CC = "g++"
PROJECT = hough_lines

SRC = main.cpp

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)

clean :
	rm -rf *.o $(PROJECT)