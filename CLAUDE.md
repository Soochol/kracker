# KRACKER - AI Video Generation Platform

## Project Overview
- **Name**: KRACKER
- **Level**: Dynamic (bkit v1.4.7)
- **Stack**: Next.js 14+ / TypeScript / Tailwind CSS / bkend.ai BaaS
- **Description**: AI-powered video generation platform with LTX-2 integration

## Tech Stack

### Frontend
- Next.js 14+ (App Router)
- TypeScript
- Tailwind CSS
- TanStack Query (data fetching)
- Zustand (state management)

### Backend (BaaS)
- bkend.ai
  - Auto REST API
  - MongoDB database
  - Built-in authentication (JWT)
  - Real-time features (WebSocket)

### AI/ML Integration
- LTX-2 Video Generation (AMD ROCm)
- Python FastAPI (local processing server)

### Deployment
- Vercel (frontend)
- bkend.ai (backend)

## Project Structure
```
kracker/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── (auth)/            # Auth routes
│   │   ├── (main)/            # Main app routes
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/             # UI components
│   │   ├── ui/                # Basic UI
│   │   └── features/          # Feature components
│   ├── hooks/                  # Custom hooks
│   ├── lib/                    # Utilities
│   │   ├── bkend.ts           # bkend.ai client
│   │   └── utils.ts
│   ├── stores/                 # Zustand stores
│   └── types/                  # TypeScript types
├── test-ltx/                   # LTX-2 video generation (Python)
├── docs/                       # PDCA documents
└── public/                     # Static assets
```

## Conventions

### Naming
- Components: PascalCase (`VideoPlayer.tsx`)
- Hooks: camelCase with 'use' prefix (`useAuth.ts`)
- Utils: camelCase (`formatDate.ts`)
- Types: PascalCase (`User.ts`)

### File Organization
- One component per file
- Co-locate tests with source files
- Group by feature, not by type

### Code Style
- Use TypeScript strict mode
- Prefer functional components
- Use Tailwind CSS for styling
- No inline styles

## SoR (Source of Record) Priority
1. **Codebase** - Implementation is the truth
2. **CLAUDE.md** - Project conventions
3. **docs/** - Design documents (reference only)

## Upgrade Path
When scaling beyond BaaS limits:
→ Enterprise Level (Microservices + K8s + Terraform)
