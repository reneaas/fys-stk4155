all: compile link run

compile:
	g++-10 -c $(wildcard *.cpp) -Ofast -march=native

link:
	g++-10 -o main.out $(wildcard *.o)

run:
	./main.out
