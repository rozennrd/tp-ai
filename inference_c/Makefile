CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11

TARGET = cnn_infer

SRCS = main_cnn.c neural_network_cnn.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
