# 🚀 LitLens Deployment Guide

## Deployment Strategy: GitHub → Vercel

### Step 1: Prepare for GitHub Push

1. **Initialize Git Repository**
```bash
cd /path/to/consolidated-litlens
git init
git add .
git commit -m "feat: consolidated and refactored LitLens v2.0

- Merged litlens + litlens-portal repositories
- Refactored all backend modules with enterprise features
- Enhanced Next.js frontend with better UX
- Added comprehensive error handling and validation
- Implemented caching, monitoring, and security features
- Maintained 100% backward compatibility"
```

2. **Create GitHub Repository**
```bash
# Option A: Create via GitHub CLI (if installed)
gh repo create litlens-v2 --public --description "LitLens v2.0 - Consolidated and refactored AI literature review assistant"

# Option B: Create manually on GitHub.com
# Then connect:
git remote add origin https://github.com/ashley-perkins/litlens-v2.git
git branch -M main
git push -u origin main
```

### Step 2: Vercel Deployment Setup

#### Backend Deployment (Vercel)

1. **Create vercel.json for backend**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "backend/app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "backend/app.py"
    },
    {
      "src": "/docs",
      "dest": "backend/app.py"
    },
    {
      "src": "/openapi.json",
      "dest": "backend/app.py"
    }
  ],
  "env": {
    "OPENAI_API_KEY": "@openai_api_key",
    "HUGGINGFACE_API_TOKEN": "@huggingface_api_token"
  }
}
```

#### Frontend Deployment (Vercel)

1. **Update package.json for Vercel**
```json
{
  "name": "litlens-frontend",
  "version": "2.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
```

2. **Create next.config.js for API routing**
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.NEXT_PUBLIC_API_URL + '/:path*',
      },
    ]
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
}

module.exports = nextConfig
```

### Step 3: Environment Variables Setup

#### For Local Development (.env.local)
```bash
# Frontend
NEXT_PUBLIC_API_URL=http://localhost:7860

# Backend (.env)
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_api_token_here
ENVIRONMENT=development
```

#### For Vercel Production
Set these in Vercel dashboard:
- `OPENAI_API_KEY` (sensitive)
- `HUGGINGFACE_API_TOKEN` (sensitive)
- `NEXT_PUBLIC_API_URL` (your backend URL)
- `ENVIRONMENT=production`

### Step 4: Deployment Process

#### Option A: Separate Frontend + Backend Deployments

1. **Deploy Backend to Vercel**
```bash
# From root directory
vercel --prod
# Select backend/ as root directory
```

2. **Deploy Frontend to Vercel**
```bash
# From frontend/ directory
cd frontend
vercel --prod
```

#### Option B: Monorepo Deployment (Recommended)

1. **Create vercel.json for monorepo**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "frontend/package.json",
      "use": "@vercel/next"
    },
    {
      "src": "backend/app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "backend/app.py"
    },
    {
      "src": "/(.*)",
      "dest": "frontend/$1"
    }
  ]
}
```

2. **Deploy to Vercel**
```bash
vercel --prod
```

### Step 5: Testing & Rollback Strategy

#### Testing Process
1. **Deploy to preview** first:
```bash
vercel  # Creates preview deployment
```

2. **Test preview URL** thoroughly
3. **Promote to production** when ready:
```bash
vercel --prod
```

#### Rollback Strategy
1. **Via Vercel Dashboard**: 
   - Go to Deployments
   - Click "Promote to Production" on previous working deployment

2. **Via Git**:
```bash
git revert HEAD  # Revert last commit
git push origin main  # Triggers new deployment
```

3. **Branch-based rollback**:
```bash
git checkout -b rollback-v1
git reset --hard [previous-working-commit]
git push origin rollback-v1
# Deploy rollback-v1 branch in Vercel
```

### Step 6: Continuous Deployment Setup

1. **Connect GitHub to Vercel**
   - Link repository in Vercel dashboard
   - Enable automatic deployments
   - Set up preview deployments for PRs

2. **Branch Strategy**
   - `main` → Production deployment
   - `develop` → Staging deployment  
   - Feature branches → Preview deployments

### Step 7: Monitoring & Health Checks

#### Health Check Endpoints
- `GET /` - Backend health
- `GET /api/health` - Detailed health status

#### Monitoring Setup
1. **Vercel Analytics** (built-in)
2. **Custom monitoring** via health endpoints
3. **Error tracking** via logs

### Quick Commands Summary

```bash
# Local development
npm run dev  # Frontend (port 3000)
uvicorn backend.app:app --reload --port 7860  # Backend

# Deploy to preview
vercel

# Deploy to production
vercel --prod

# Rollback via Git
git revert HEAD && git push

# Environment variables
vercel env add OPENAI_API_KEY
vercel env add HUGGINGFACE_API_TOKEN
```

This setup gives you:
- **Instant rollbacks** 
- **Preview deployments** for testing
- **Automatic deployments** on push
- **Environment isolation**
- **Easy scaling** and monitoring