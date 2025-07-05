# 🧠📚 LitLens - AI-Powered Literature Review Assistant

A lightweight AI-powered literature review assistant designed to summarize, structure, and surface scientific insights with clarity and speed.

## 🚀 Features

- **Multi-PDF Processing**: Upload and analyze multiple scientific PDFs simultaneously
- **AI Summarization**: Generate structured summaries using OpenAI GPT-4 or Hugging Face models
- **Semantic Filtering**: Use embeddings to match papers with your research goals
- **Report Generation**: Download comprehensive Markdown reports
- **Modern UI**: Beautiful, responsive Next.js frontend with drag-and-drop support
- **Flexible Deployment**: Docker support with local development options

## 🏗️ Architecture

```
LitLens/
├── backend/                 # FastAPI backend
│   ├── api/                # API routes and models
│   ├── modules/            # Core processing modules
│   └── utils/              # Utility functions
├── frontend/               # Next.js frontend
│   ├── app/               # Next.js app directory
│   ├── components/        # React components
│   └── lib/               # Utilities
└── requirements.txt       # Python dependencies
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 18+
- OpenAI API Key (optional, for enhanced features)
- Hugging Face API Token

### Backend Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key_here
# HUGGINGFACE_API_TOKEN=your_hf_token_here
```

3. **Run Backend Server**
```bash
uvicorn backend.app:app --reload --port 7860
```

### Frontend Setup

1. **Navigate to Frontend**
```bash
cd frontend
```

2. **Install Dependencies**
```bash
npm install
# or
pnpm install
```

3. **Environment Configuration**
```bash
cp .env.example .env.local
# For local development, the default should work:
# NEXT_PUBLIC_API_URL=http://localhost:7860
```

4. **Run Development Server**
```bash
npm run dev
# or
pnpm dev
```

### Full Local Development

1. **Terminal 1 - Backend**
```bash
uvicorn backend.app:app --reload --port 7860
```

2. **Terminal 2 - Frontend**
```bash
cd frontend && npm run dev
```

3. **Access Application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:7860
- API Documentation: http://localhost:7860/docs

## 📡 API Endpoints

### Core Endpoints
- `POST /summarize-pdfs` - Process multiple PDFs with OpenAI (primary)
- `POST /summarize-hf-pdfs` - Process PDFs with Hugging Face models
- `POST /summarize-hf` - Summarize text with Hugging Face
- `POST /report` - Generate downloadable Markdown reports
- `GET /download` - Download generated files

### Development
- `GET /` - Health check
- `GET /docs` - Interactive API documentation

## 🔧 CLI Usage

Process PDFs directly via command line:

```bash
python -m backend.litlens \
  --goal "Identify biomarkers for appendiceal neoplasms" \
  --input-dir path/to/pdfs \
  --output reports \
  --format md
```

## 🐳 Docker Deployment

```bash
# Build container
docker build -t litlens .

# Run with environment variables
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e HUGGINGFACE_API_TOKEN=your_token \
  litlens
```

## 🧪 Development Notes

### AI Model Configuration

- **OpenAI Models**: GPT-4 for summarization, Ada-002 for embeddings
- **Hugging Face Models**: 
  - Summarization: `philschmid/bart-large-cnn-samsum`
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

### File Processing Pipeline

1. **PDF Extraction**: Extract text using pdfminer.six
2. **Text Chunking**: Intelligent section-based chunking
3. **Embedding**: Generate semantic embeddings for goal matching
4. **Filtering**: Cosine similarity filtering for relevance
5. **Summarization**: AI-powered summarization with goal context
6. **Report Generation**: Structured Markdown output

## 🚀 Live Deployment

LitLens is live and hosted on Hugging Face Spaces:
- **Demo**: https://ashley-perkins-litlens.hf.space
- **API Docs**: https://ashley-perkins-litlens.hf.space/docs

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
  ## Deployment Status
  ✅ Configured for Vercel deployment
  ✅ Environment variables set in dashboard
  ✅ Ready for production use
