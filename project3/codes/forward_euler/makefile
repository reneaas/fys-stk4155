all: compile run plot

compile:
	g++-10 -o main.exe $(wildcard *.cpp) -O3

run:
	./main.exe u_dx_0.01_t_1.2.txt 1.2 0.01

plot:
	python3 main.py
