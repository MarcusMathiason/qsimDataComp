include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/iterator
INCS = -I../include
LIBS = -L../lib -lzfp
CLIBS = $(LIBS) -lm
CXXLIBS = $(LIBS)

all: $(TARGETS)

$(BINDIR)/iterator: iterator.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) -I../array iterator.cpp $(CXXLIBS) -o $@
clean:
	rm -f $(TARGETS) diffusionC.o
