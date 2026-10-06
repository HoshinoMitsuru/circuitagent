# Circuit Agent

Circuit Intelligent Analysis Agent — A one-stop solution from circuit diagram photos, SVG, or KiCad files to complete analysis results.

## Features

### Multiple Input Methods
- **Image OCR Recognition**: Automatically identify circuit components and connections from photos/screenshots
- **SVG Parsing**: Supports importing circuit diagrams from SVG
- **KiCad Netlist**: Supports KiCad schematics and netlist import
- **SPICE Netlist**: Supports reading and exporting SPICE netlists

### Circuit Solving
- **Node Voltage Method (MNA)**: Classic matrix solving
- **Branch Current Method**: Solving based on fundamental loops
- **DC Simplification**: Automatically merge series/parallel components
- **ngspice Solver**: Integrated open-source SPICE engine
- **Triple-Method Cross-Verification**: Cross-validate results using three methods

### Controlled Sources and Parameter System
- Voltage Controlled Voltage Source (VCVS/E)
- Voltage Controlled Current Source (VCCS/G)
- Current Controlled Voltage Source (CCVS/H)
- Current Controlled Current Source (CCCS/F)
- Parameter Expression Support

### Verification Functions
- KCL (Kirchhoff's Current Law) Verification
- KVL (Kirchhoff's Voltage Law) Verification
- Power Balance Verification

### Visualization
- SVG Rendered Circuit Diagram
- Display Node Voltages and Branch Currents
- Dark/Light Theme Switching

## Installation

### Environment Requirements
- Python 3.10+
- Windows 10/11 (Supports packaging into single-file exe)

### Local Run

```bash
# Clone repository
git clone https://gitee.com/psycheclaritas/circuit_agent.git
cd circuit_agent

# Install dependencies
pip install -e .

# Start Web Service
python -m app.api.server
```

After service starts, access http://127.0.0.1:8765

### Package as Single-file EXE

```bash
python tools/build_exe.py
```

After packaging is complete, the executable is located at `dist/circuit_agent.exe`, with the ngspice engine built-in.

## Usage

### Web Interface

Open your browser to access after starting the service:
```
http://127.0.0.1:8765
```

Supports:
- Drag and drop upload of circuit images/SVG files
- OCR automatic component recognition
- Manually edit component parameters
- View solving results and verification information

### API Interfaces

#### Health Check
```bash
GET /api/health
```

#### Import Circuit
```bash
POST /api/import
Content-Type: multipart/form-data

file: <circuit file>
tol: 6.0  # Topology tolerance
ref_node: "0"  # Reference node (optional)
```

#### Solve Circuit
```bash
POST /api/solve
Content-Type: application/json

{
  "method": "mna",  # mna | branch | ngspice | auto
  "verify": true
}
```

#### Render SVG
```bash
POST /api/render
Content-Type: application/json

{
  "mode": "auto",  # auto | grid | radial
  "dark": false
}
```

#### SPICE Interconversion
```bash
POST /api/from-spice
Content-Type: application/json

{
  "netlist": "* Sample circuit\nV1 1 0 10\nR1 1 2 1k\nR2 2 0 2k"
}
```

### MCP Server (LLM Integration)

Supported as an MCP tool for Large Language Model invocation:

```bash
python mcp_server/server.py
```

Provides three tools:
- `solve`: Solve circuits
- `validate`: Verify solving results
- `elements`: Get circuit component list

## Project Structure

```
circuit_agent/
├── app/
│   ├── api/          # FastAPI Web Service
│   ├── ingest/       # Circuit Input Parsing
│   │   ├── kicad_in.py    # KiCad Netlist Parsing
│   │   ├── svg_in.py      # SVG Parsing
│   │   ├── topology.py    # Topology Analysis
│   │   └── values.py      # Parameter Parsing
│   ├── ir/           # Internal Circuit Representation
│   │   ├── model.py       # Data Model
│   │   ├── params.py      # Parameter System
│   │   ├── render.py      # SVG Rendering
│   │   ├── spice.py       # SPICE Interconversion
│   │   └── probes.py      # Probe Handling
│   ├── solver/       # Solvers
│   │   ├── mna.py         # Node Voltage Method
│   │   ├── branch.py      # Branch Current Method
│   │   ├── dc_reduce.py   # DC Simplification
│   │   ├── controlled.py  # Controlled Source Handling
│   │   ├── ngspice.py     # ngspice Interface
│   │   └── reconcile.py   # Result Reconciliation
│   └── vision/       # Computer Vision
│       ├── ocr.py         # OCR Recognition
│       ├── symbols.py     # Component Detection
│       ├── wires.py       # Wire Extraction
│       └── pipeline.py    # Processing Pipeline
├── tests/            # Test Suite
├── tools/            # Build Tools
├── web/              # Frontend Pages
└── mcp_server/       # MCP Service
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test individual modules
python tests/test_solver.py
python tests/test_controlled.py
python tests/test_vision.py
python tests/test_api.py
```

## Configuration

Configuration file is located at `config/secrets.json` (needs to be created from example):

```json
{
  "vlm_api_key": "your-api-key",
  "vlm_endpoint": "https://api.example.com/vision"
}
```

OCR configuration can be modified via Web Interface or API.

## License

MIT License - See LICENSE file for details.