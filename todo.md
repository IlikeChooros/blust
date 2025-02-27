
TODO:
- [ ] In cuda code, use `throw` instead of `exit` to handle errors.
- [ ] Add more tests:
	- [x] Test all operations with `cuda` backend (against cpu).
- [x] Make optimized backend default, and make it discover if cuda device is available, else use cpu.
- [x] Optimize `cuda` backend (rn 175ms) -> (127ms - on debug, 64ms) stop using `cudaMalloc` and `cudaMemcpy` for every operation, instead just check if the previous size is enough, if not, allocate new memory.