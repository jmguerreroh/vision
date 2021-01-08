CC = "g++"

PROJECT1 = wavelets_haar
PROJECT2 = wavelets_mahavir

SRC1 = main_haar.cpp
SRC2 = main_mahavir.cpp

LIBS = `pkg-config opencv4 --cflags --libs`

all : $(PROJECT1) $(PROJECT2)

$(PROJECT1) : $(SRC1)
	$(CC) $(SRC1) -o $(PROJECT1) $(LIBS)
	
$(PROJECT2) : $(SRC2)
	$(CC) $(SRC2) -o $(PROJECT2) $(LIBS)

clean :
	rm -rf *.o $(PROJECT1) $(PROJECT2)