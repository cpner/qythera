module.exports = { output: 'standalone', async rewrites() { return [{source:'/v1/:path*', destination:'http://localhost:8000/v1/:path*'}]; } };
