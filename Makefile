.POSIX:

dataset:
	@python3 scripts/download_dataset.py

report:
	$(MAKE) -C report

clean:
	$(MAKE) clean -C report

.PHONY: dataset report clean
