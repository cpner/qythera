# Qythera Edge (WebAssembly)

Run Qythera models directly in the browser using WebAssembly.

## Features

- Runs entirely in-browser (no server needed for small models)
- WebGPU acceleration support
- Works offline with service workers
- Compatible with all modern browsers

## Usage

```javascript
import { QytheraWASM } from '@qythera/edge-wasm';
const model = await QytheraWASM.load('vaelon-1b');
const response = await model.generate('Hello!');
```
