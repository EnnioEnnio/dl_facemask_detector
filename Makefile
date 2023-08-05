.POSIX:

dataset:
	@python3 ./util.py dataset

model:
	@python3 ./util.py model

report:
	$(MAKE) -C report

clean:
	$(MAKE) clean -C report

format:
	@black .

.PHONY: dataset report clean
