.POSIX:

report:
	$(MAKE) -C report

clean:
	$(MAKE) clean -C report

.PHONY: report clean
