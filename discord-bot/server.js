import express from 'express';
import helmet from 'helmet';
import { RateLimiterMemory } from 'rate-limiter-flexible';

const server = express();

// Security middleware
server.use(helmet());

// Rate limiting
const rateLimiter = new RateLimiterMemory({
  keyPrefix: 'middleware',
  points: 100, // Number of requests
  duration: 60, // Per 60 seconds
});

server.use(async (req, res, next) => {
  try {
    await rateLimiter.consume(req.ip);
    next();
  } catch (rejRes) {
    res.status(429).send('Too Many Requests');
  }
});

// Health check endpoint
server.get("/", (req, res) => {
  res.json({
    status: "running",
    message: "Zhongli Discord Bot is active",
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Health endpoint for monitoring
server.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    version: process.env.npm_package_version || "2.0.0"
  });
});

function keepAlive() {
  const port = process.env.PORT || 3000;

  server.listen(port, () => {
    console.log(`🚀 Server is ready on port ${port}`);
    console.log(`🔗 Health check: http://localhost:${port}/health`);
  });

  // Graceful shutdown
  process.on('SIGTERM', () => {
    console.log('🔄 SIGTERM received, shutting down gracefully');
    server.close(() => {
      console.log('✅ Server closed');
      process.exit(0);
    });
  });
}

export default keepAlive;
