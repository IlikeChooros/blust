
TODO:
- [ ] In cuda code, use `throw` instead of `exit` to handle errors.
- [ ] Add more tests:
	- [ ] Test all operations with `cuda` backend (against cpu).
- [x] Make optimized backend default, and make it discover if cuda device is available, else use cpu.