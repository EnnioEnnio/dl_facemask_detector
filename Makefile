.POSIX:

dataset:
	@python3 scripts/download_dataset.py

model:
	@python3 scripts/download_model.py

report:
	$(MAKE) -C report

clean:
	$(MAKE) clean -C report

format:
	@black .

.PHONY: dataset report clean
