/**
 * KRACKER Type Definitions
 */

// Base document type (bkend.ai auto-generated fields)
export interface BaseDocument {
  _id: string;
  createdAt: Date;
  updatedAt: Date;
}

// User types
export interface User extends BaseDocument {
  email: string;
  name?: string;
  avatarUrl?: string;
}

// Video generation types
export interface VideoGeneration extends BaseDocument {
  userId: string;
  prompt: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  videoUrl?: string;
  thumbnailUrl?: string;
  duration?: number;
  settings: VideoSettings;
  error?: string;
}

export interface VideoSettings {
  width: number;
  height: number;
  fps: number;
  numFrames: number;
  seed?: number;
}

// API response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
  };
}

// Pagination types
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}
