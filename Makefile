python=python3
test_args=-vv -s -o log_cli=true --log-cli-level=INFO

test:
	${python} -m pytest ${test_args} tests
