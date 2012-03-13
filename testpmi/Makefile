CPPFLAGS += -I../mpich2/src/include -MMD
CFLAGS += -g -Wall -W -Wno-unused -O2 -std=c99
LDFLAGS += ${LIBDIRS} -lpopt -lm $(LDFLAGS_$@)

CC=mpicc


all:


PROGS		+=pmi_hw
# LDFLAGS_pmi_hw	+= -L/opt/parastation/lib -lpscom -L/opt/parastation/mpi2/lib -lmpich
pmi_hw: pmi_hw.o


all: $(PROGS)


DEPFILES := $(wildcard *.d) dummy.d
-include $(DEPFILES)

clean:
	$(RM) *.o ${PROGS} ${DEPFILES}