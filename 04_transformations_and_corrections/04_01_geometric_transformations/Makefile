CC = "g++"
PROJECT = transformation

SRC1 = resize
SRC2 = rotation
SRC3 = translation
SRC4 = inclination

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC1).cpp $(SRC2).cpp $(SRC3).cpp $(SRC4).cpp
	$(CC) $(SRC1).cpp -o $(SRC1) $(LIBS)
	$(CC) $(SRC2).cpp -o $(SRC2) $(LIBS)
	$(CC) $(SRC3).cpp -o $(SRC3) $(LIBS)
	$(CC) $(SRC4).cpp -o $(SRC4) $(LIBS)

clean :
	rm -rf *.o resize rotation translation inclination