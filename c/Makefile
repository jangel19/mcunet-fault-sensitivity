# build the tests and trial runner
# make test also writes fresh python reference outputs
# tflite is an external install, see build_tflite.md

TFLITE ?= $(HOME)/.local/tflite-2.21.0
MODEL  ?= $(if $(MCUNET_MODEL),$(MCUNET_MODEL),$(HOME)/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite)
SUBSET ?= ../subset
CSV    ?= ../layer_inventory.csv
PYTHON ?= ../venv311/bin/python
SMOKE_LIMIT ?= 30

CC      = cc
CFLAGS  = -std=c11 -O2 -g -Wall -Wextra -I"$(TFLITE)/include"
LDFLAGS = -L"$(TFLITE)/lib" -ltensorflowlite_c -Wl,-rpath,"$(TFLITE)/lib"

all: test_fi run_trials

test_fi: test_fi.c fault_inject.c fault_inject.h
	$(CC) $(CFLAGS) -o $@ test_fi.c fault_inject.c $(LDFLAGS)

run_trials: run_trials.c fault_inject.c fault_inject.h
	$(CC) $(CFLAGS) -o $@ run_trials.c fault_inject.c $(LDFLAGS)

test: test_fi
	"$(PYTHON)" ref_logits.py --subset "$(SUBSET)" --k 200 --model "$(MODEL)"
	./test_fi --model "$(MODEL)" --subset "$(SUBSET)" --csv "$(CSV)" --images 200

smoke: run_trials
	./run_trials --model "$(MODEL)" --subset "$(SUBSET)" --mode corner --layers 1,62,120 --bits 0,7 --limit $(SMOKE_LIMIT) --images 20 --out trials_smoke.csv

clean:
	rm -rf test_fi run_trials *.dSYM trials*.csv

.PHONY: all test smoke clean
