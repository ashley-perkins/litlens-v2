# Environment Setup Guide

## Local Development

### Required Files:
- `.env.local` (already created, minimal setup)

### Environment Variables:
**None required for basic functionality**
- The app uses internal Next.js API routes
- No external API keys needed for placeholder functionality

### Optional Environment Variables:
```bash
# Only if you want to enable full AI functionality
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
```

## Production Deployment (Vercel)

### Vercel Configuration:
1. **Build Command**: `npm run build` (default)
2. **Output Directory**: `.next` (default)  
3. **Install Command**: `npm install` (default)
4. **Root Directory**: `/` (root of repo, NOT /frontend)

### Environment Variables in Vercel:
**None required for current functionality**

### Optional Production Variables:
```bash
# Only add these if implementing full AI features
OPENAI_API_KEY=your_production_openai_key
HUGGINGFACE_API_KEY=your_production_hf_key
```

## Verification Steps

### After Deployment:
1. Visit your Vercel URL
2. Upload a PDF file
3. Enter a research goal
4. Click "Summarize"
5. Verify you get a structured response

### Expected Response:
```json
{
  "goal": "your research goal",
  "summaries": [
    {
      "filename": "your-file.pdf",
      "title": "Analysis of your-file.pdf", 
      "summary": "Detailed placeholder summary with file info..."
    }
  ],
  "status": "success",
  "message": "PDF processing completed successfully"
}
```

## Troubleshooting

### Common Issues:
1. **404 on API routes**: Ensure building from root directory
2. **Component not found**: Check imports use `/components/` not `/frontend/components/`
3. **Build errors**: Verify no duplicate files in `/frontend/` directory

### Debug Steps:
1. Check Vercel build logs for errors
2. Test API endpoints directly: `/api/health` and `/api/summarize-pdfs`
3. Verify no TypeScript or ESLint blocking errors
4. Check browser console for client-side errors

## Development Commands

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Test API endpoints
node test-api.js
```

## File Structure

```
/
├── app/
│   ├── api/
│   │   ├── health/route.ts
│   │   └── summarize-pdfs/route.ts
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── lit-lens-uploader.tsx
│   └── ui/
├── next.config.mjs
├── package.json
└── .env.local
```

## Notes

- No external dependencies on HuggingFace Spaces
- All functionality self-contained in Next.js app
- API routes handle both development and production
- Scalable architecture for adding real AI features later