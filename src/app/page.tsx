import Link from 'next/link';

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="max-w-4xl text-center">
        <h1 className="text-5xl font-bold mb-6">
          KRACKER
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
          AI-Powered Video Generation with LTX-2
        </p>

        <div className="flex gap-4 justify-center">
          <Link
            href="/login"
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Get Started
          </Link>
          <Link
            href="/dashboard"
            className="px-6 py-3 border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            Dashboard
          </Link>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="p-6 border border-gray-200 dark:border-gray-800 rounded-xl">
            <h3 className="text-lg font-semibold mb-2">Text to Video</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Describe your vision and watch it come to life
            </p>
          </div>
          <div className="p-6 border border-gray-200 dark:border-gray-800 rounded-xl">
            <h3 className="text-lg font-semibold mb-2">Image to Video</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Animate your images with AI-powered motion
            </p>
          </div>
          <div className="p-6 border border-gray-200 dark:border-gray-800 rounded-xl">
            <h3 className="text-lg font-semibold mb-2">AMD ROCm</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Optimized for AMD GPUs with ROCm acceleration
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
