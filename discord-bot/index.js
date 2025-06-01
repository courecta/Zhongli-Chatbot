import dotenv from 'dotenv';
import path from 'path';
dotenv.config({ path: path.resolve(__dirname, '../.env') });
import 'dotenv/config';
import { Client, GatewayIntentBits, Events, Partials } from 'discord.js';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import { keepAlive } from './server.js';

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent
  ]
});

// Rate limiting for API calls
const apiRateLimiter = new RateLimiterMemory({
  keyPrefix: 'api_calls',
  points: 10, // Number of API calls
  duration: 60, // Per 60 seconds per user
});

// Rate limiting for user messages
const userRateLimiter = new RateLimiterMemory({
  keyPrefix: 'user_messages',
  points: 5, // Number of messages
  duration: 30, // Per 30 seconds per user
});

let responding = true; // Default to true, managed in-memory

const HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium";

// Modern Hugging Face API configuration
const API_CONFIG = {
  url: 'https://api-inference.huggingface.co/models/Gappy/DialoGPT-small-Zhongli',
  maxRetries: 3,
  timeout: 30000, // 30 seconds
  defaultParams: {
    max_length: 512,
    temperature: 0.7,
    do_sample: true,
    top_p: 0.9
  }
};

// Conversation context management
const conversationContexts = new Map();
const MAX_CONTEXT_LENGTH = 5;

/**
 * Store conversation context for better responses
 */
function updateConversationContext(userId, userMessage, botResponse) {
  if (!conversationContexts.has(userId)) {
    conversationContexts.set(userId, []);
  }

  const context = conversationContexts.get(userId);
  context.push({ user: userMessage, bot: botResponse });

  // Keep only recent messages
  if (context.length > MAX_CONTEXT_LENGTH) {
    context.shift();
  }
}

/**
 * Get conversation context for user
 */
function getConversationContext(userId) {
  return conversationContexts.get(userId) || [];
}

/**
 * Modern Hugging Face API call with error handling and retries
 */
async function callHuggingFaceAPI(input, userId, retryCount = 0) {
  try {
    // Build context-aware input
    const context = getConversationContext(userId);
    let contextualInput = input;

    if (context.length > 0) {
      const recentContext = context.slice(-2); // Last 2 exchanges
      const contextString = recentContext
        .map(ctx => `User: ${ctx.user}\nZhongli: ${ctx.bot}`)
        .join('\n');
      contextualInput = `${contextString}\nUser: ${input}\nZhongli:`;
    }

    const payload = {
      inputs: contextualInput,
      parameters: API_CONFIG.defaultParams,
      options: {
        wait_for_model: true,
        use_cache: false
      }
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);

    const response = await fetch(API_CONFIG.url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.HUGGINGFACE_TOKEN}`,
        'Content-Type': 'application/json',
        'User-Agent': 'ZhongliBot/2.0'
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    // Handle different response formats
    if (Array.isArray(data) && data[0]?.generated_text) {
      let generatedText = data[0].generated_text;

      // Extract only the new response (after the last "Zhongli:")
      const lastZhongliIndex = generatedText.lastIndexOf('Zhongli:');
      if (lastZhongliIndex !== -1) {
        generatedText = generatedText.substring(lastZhongliIndex + 8).trim();
      }

      // Clean up the response
      generatedText = generatedText
        .split('\n')[0] // Take only first line
        .replace(/^[^a-zA-Z]*/, '') // Remove leading non-letters
        .trim();

      return generatedText || "I understand your words, though I find myself at a loss for a proper response.";
    }

    if (data.error) {
      throw new Error(`API Error: ${data.error}`);
    }

    throw new Error('Unexpected API response format');

  } catch (error) {
    console.error(`API call failed (attempt ${retryCount + 1}):`, error.message);

    // Retry logic
    if (retryCount < API_CONFIG.maxRetries && !error.name?.includes('Abort')) {
      const delay = Math.pow(2, retryCount) * 1000; // Exponential backoff
      await new Promise(resolve => setTimeout(resolve, delay));
      return callHuggingFaceAPI(input, userId, retryCount + 1);
    }

    // Fallback responses for different error types
    if (error.name?.includes('Abort')) {
      return "My thoughts require more time to process. Perhaps try again in a moment.";
    }

    if (error.message.includes('503') || error.message.includes('502')) {
      return "The winds of time are turbulent. The model is currently loading. Please try again shortly.";
    }

    return "Even I, with my vast experience, sometimes find myself contemplating in silence.";
  }
}

/**
 * Create modern embed for responses
 */
function createResponseEmbed(userMessage, botResponse, isError = false) {
  const embed = new EmbedBuilder()
    .setColor(isError ? 0xff0000 : 0xd4af37) // Red for errors, gold for normal
    .setAuthor({
      name: 'Zhongli',
      iconURL: 'https://i.imgur.com/YourZhongliIcon.png' // Add your icon URL
    })
    .addFields(
      { name: 'Your Message', value: userMessage.length > 1024 ? userMessage.substring(0, 1021) + '...' : userMessage },
      { name: 'Response', value: botResponse.length > 1024 ? botResponse.substring(0, 1021) + '...' : botResponse }
    )
    .setTimestamp()
    .setFooter({ text: 'Zhongli • Geo Archon' });

  return embed;
}

// Bot ready event
client.once(Events.ClientReady, async () => {
  console.log(`✅ Logged in as ${client.user.tag}`);
  console.log(`🏛️ Zhongli is ready to share wisdom in ${client.guilds.cache.size} servers`);

  // Set bot status
  client.user.setActivity('the contracts of Liyue', { type: 'WATCHING' });

  console.log(`Bot is currently ${responding ? 'responding' : 'not responding'} to messages.`);
});

// Message handling with modern Discord.js v14
client.on(Events.MessageCreate, async (message) => {
  // Ignore bot messages and messages without content
  if (message.author.bot || !message.content.trim()) {
    return;
  }

  const userId = message.author.id;
  const userMessage = message.content.trim();

  try {
    // Check if bot is responding
    if (!responding) {
      return; // Bot is disabled
    }

    // Rate limiting for users
    try {
      await userRateLimiter.consume(userId);
    } catch (rejRes) {
      const embed = new EmbedBuilder()
        .setColor(0xff9900)
        .setDescription("⏳ Please wait a moment before sending another message.")
        .setFooter({ text: "Rate limited" });

      await message.reply({ embeds: [embed] });
      return;
    }

    // Handle admin commands
    if (message.content.startsWith('!zhongli')) {
      const args = message.content.split(' ').slice(1);
      const command = args[0]?.toLowerCase();

      switch (command) {
        case 'toggle':
          if (message.author.id !== 'YOUR_USER_ID' && !message.member.permissions.has("Administrator")) { // Replace YOUR_USER_ID or use permissions
            return message.reply("You don't have permission to use this command.");
          }
          const newState = !responding;
          responding = newState;
          message.reply(`Zhongli will now ${newState ? 'respond' : 'not respond'} to messages.`);
          console.log(`Responding state changed to: ${newState} by ${message.author.tag}`);
          return;

        case 'clear':
          conversationContexts.delete(userId);
          const embed = new EmbedBuilder()
            .setColor(0x00ff00)
            .setDescription("🧹 Your conversation context has been cleared.")
            .setFooter({ text: "Context cleared" });

          await message.reply({ embeds: [embed] });
          return;

        case 'stats':
          const stats = {
            servers: client.guilds.cache.size,
            users: client.users.cache.size,
            uptime: Math.floor(process.uptime()),
            memory: Math.round(process.memoryUsage().heapUsed / 1024 / 1024)
          };

          const statsEmbed = new EmbedBuilder()
            .setColor(0xd4af37)
            .setTitle("📊 Bot Statistics")
            .addFields(
              { name: "Servers", value: stats.servers.toString(), inline: true },
              { name: "Users", value: stats.users.toString(), inline: true },
              { name: "Uptime", value: `${stats.uptime}s`, inline: true },
              { name: "Memory", value: `${stats.memory}MB`, inline: true }
            )
            .setTimestamp();

          await message.reply({ embeds: [statsEmbed] });
          return;
      }
    }

    // Skip if message is a command for other bots
    if (userMessage.startsWith('!') || userMessage.startsWith('/')) {
      return;
    }

    // Rate limiting for API calls
    try {
      await apiRateLimiter.consume(userId);
    } catch (rejRes) {
      const embed = new EmbedBuilder()
        .setColor(0xff9900)
        .setDescription("⏳ You're making requests too quickly. Please wait a moment.")
        .setFooter({ text: "API rate limited" });

      await message.reply({ embeds: [embed] });
      return;
    }

    // Show typing indicator
    await message.channel.sendTyping();

    // Call AI model
    const botResponse = await callHuggingFaceAPI(userMessage, userId);

    // Update conversation context
    updateConversationContext(userId, userMessage, botResponse);

    // Create and send response embed
    const responseEmbed = createResponseEmbed(userMessage, botResponse);
    await message.reply({ embeds: [responseEmbed] });

    // Log successful interaction
    console.log(`💬 ${message.author.tag} -> ${userMessage.substring(0, 50)}${userMessage.length > 50 ? '...' : ''}`);

  } catch (error) {
    console.error('Message handling error:', error);

    const errorEmbed = new EmbedBuilder()
      .setColor(0xff0000)
      .setDescription("⚠️ An error occurred while processing your message. Please try again.")
      .setFooter({ text: "Error occurred" });

    try {
      await message.reply({ embeds: [errorEmbed] });
    } catch (replyError) {
      console.error('Failed to send error message:', replyError);
    }
  }
});

// Error handling
client.on(Events.Error, (error) => {
  console.error('Discord client error:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

// Start the server and bot
keepAlive();

// Login to Discord
client.login(process.env.DISCORD_TOKEN).catch(error => {
  console.error('Failed to login to Discord:', error);
  process.exit(1);
});
