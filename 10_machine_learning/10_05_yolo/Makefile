CC = "g++"
PROJECT = yolo

SRC = main.cpp

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) -pthread $(LIBS)

clean :
	rm -rf *.o $(PROJECT)