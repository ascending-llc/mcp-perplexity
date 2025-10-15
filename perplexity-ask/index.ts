#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import http from "http";

/**
 * Definition of the Perplexity Ask Tool.
 * This tool accepts an array of messages and returns a chat completion response
 * from the Perplexity API, with citations appended to the message if provided.
 */
const PERPLEXITY_ASK_TOOL: Tool = {
  name: "perplexity_ask",
  description:
    "Engages in a conversation using the Sonar API with the sonar-pro model. " +
    "This tool is optimized for general questions, conversations, and quick responses. " +
    "Accepts an array of messages (each with a role and content) " +
    "and returns a chat completion response from the Perplexity model with real-time information and citations.",
  inputSchema: {
    type: "object",
    properties: {
      messages: {
        type: "array",
        description: "Array of conversation messages. Each message should have a 'role' (system, user, or assistant) and 'content' (the message text).",
        items: {
          type: "object",
          properties: {
            role: {
              type: "string",
              description: "Role of the message sender (system, user, or assistant)",
              enum: ["system", "user", "assistant"]
            },
            content: {
              type: "string",
              description: "The actual content/text of the message"
            },
          },
          required: ["role", "content"],
        },
        minItems: 1
      },
    },
    required: ["messages"],
    additionalProperties: false
  },
};

/**
 * Definition of the Perplexity Research Tool.
 * This tool performs deep research queries using the Perplexity API.
 */
const PERPLEXITY_RESEARCH_TOOL: Tool = {
  name: "perplexity_research",
  description:
    "Performs comprehensive deep research using the Perplexity API with the sonar-deep-research model. " +
    "This tool is specifically designed for in-depth analysis, research queries, and complex topics that require " +
    "extensive information gathering from multiple sources. It provides more thorough responses than the standard ask tool " +
    "and includes comprehensive citations and references. Use this for academic research, detailed analysis, " +
    "market research, or when you need exhaustive information on a topic.",
  inputSchema: {
    type: "object",
    properties: {
      messages: {
        type: "array",
        description: "Array of conversation messages for research context. Typically should include a detailed research query or question.",
        items: {
          type: "object",
          properties: {
            role: {
              type: "string",
              description: "Role of the message sender (system, user, or assistant)",
              enum: ["system", "user", "assistant"]
            },
            content: {
              type: "string",
              description: "The research query or context. Be specific about what you want to research."
            },
          },
          required: ["role", "content"],
        },
        minItems: 1
      },
    },
    required: ["messages"],
    additionalProperties: false
  },
};

/**
 * Definition of the Perplexity Reason Tool.
 * This tool performs reasoning queries using the Perplexity API.
 */
const PERPLEXITY_REASON_TOOL: Tool = {
  name: "perplexity_reason",
  description:
    "Performs advanced reasoning and analytical tasks using the Perplexity API with the sonar-reasoning-pro model. " +
    "This tool is optimized for logical reasoning, problem-solving, mathematical computations, code analysis, " +
    "step-by-step thinking, and complex analytical tasks. It provides structured, well-reasoned responses " +
    "with clear logical progression. Use this tool for: mathematical problems, logical puzzles, code debugging, " +
    "analytical thinking, decision-making processes, or any task requiring systematic reasoning.",
  inputSchema: {
    type: "object",
    properties: {
      messages: {
        type: "array",
        description: "Array of conversation messages presenting the reasoning task or problem to solve.",
        items: {
          type: "object",
          properties: {
            role: {
              type: "string",
              description: "Role of the message sender (system, user, or assistant)",
              enum: ["system", "user", "assistant"]
            },
            content: {
              type: "string",
              description: "The problem, question, or task that requires reasoning. Be clear about what kind of reasoning or analysis is needed."
            },
          },
          required: ["role", "content"],
        },
        minItems: 1
      },
    },
    required: ["messages"],
    additionalProperties: false
  },
};

// Retrieve the Perplexity API key from environment variables
const PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY;
if (!PERPLEXITY_API_KEY) {
  console.error("Error: PERPLEXITY_API_KEY environment variable is required");
  process.exit(1);
}

// Get the port from environment variables (default to 3000)
const PORT = parseInt(process.env.PERPLEXITY_PORT || "3334", 10);
const DEBUG = process.env.DEBUG === "true";

// Check if we should run in HTTP mode
const HTTP_MODE = process.env.HTTP_MODE === "true" || process.argv.includes("--http");

/**
 * Enhanced logging function
 */
function log(level: 'INFO' | 'ERROR' | 'DEBUG', message: string, data?: any) {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] [${level}] ${message}`;
  
  if (level === 'ERROR') {
    console.error(logMessage, data ? JSON.stringify(data, null, 2) : '');
  } else if (level === 'DEBUG' && DEBUG) {
    console.error(logMessage, data ? JSON.stringify(data, null, 2) : '');
  } else if (level === 'INFO') {
    console.error(logMessage, data ? JSON.stringify(data, null, 2) : '');
  }
}

/**
 * Performs a chat completion by sending a request to the Perplexity API.
 * Appends citations to the returned message content if they exist.
 *
 * @param {Array<{ role: string; content: string }>} messages - An array of message objects.
 * @param {string} model - The model to use for the completion.
 * @returns {Promise<string>} The chat completion result with appended citations.
 * @throws Will throw an error if the API request fails.
 */
async function performChatCompletion(
  messages: Array<{ role: string; content: string }>,
  model: string = "sonar-pro"
): Promise<string> {
  log('INFO', `Starting chat completion with model: ${model}`);
  log('DEBUG', 'Messages sent to API', { messages, model });

  // Construct the API endpoint URL and request body
  const url = new URL("https://api.perplexity.ai/chat/completions");
  const body = {
    model: model, // Model identifier passed as parameter
    messages: messages,
    // Additional parameters can be added here if required (e.g., max_tokens, temperature, etc.)
    // See the Sonar API documentation for more details: 
    // https://docs.perplexity.ai/api-reference/chat-completions
  };

  let response;
  try {
    log('DEBUG', 'Sending request to Perplexity API', { url: url.toString(), body });
    response = await fetch(url.toString(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${PERPLEXITY_API_KEY}`,
      },
      body: JSON.stringify(body),
    });
  } catch (error) {
    log('ERROR', 'Network error while calling Perplexity API', error);
    throw new Error(`Network error while calling Perplexity API: ${error}`);
  }

  // Check for non-successful HTTP status
  if (!response.ok) {
    let errorText;
    try {
      errorText = await response.text();
    } catch (parseError) {
      errorText = "Unable to parse error response";
    }
    log('ERROR', `Perplexity API HTTP error: ${response.status}`, { status: response.status, statusText: response.statusText, errorText });
    throw new Error(
      `Perplexity API error: ${response.status} ${response.statusText}\n${errorText}`
    );
  }

  // Attempt to parse the JSON response from the API
  let data;
  try {
    data = await response.json();
    log('DEBUG', 'Received response from Perplexity API', { data });
  } catch (jsonError) {
    log('ERROR', 'Failed to parse JSON response from Perplexity API', jsonError);
    throw new Error(`Failed to parse JSON response from Perplexity API: ${jsonError}`);
  }

  // Directly retrieve the main message content from the response 
  let messageContent = data.choices[0].message.content;

  // If citations are provided, append them to the message content
  if (data.citations && Array.isArray(data.citations) && data.citations.length > 0) {
    log('DEBUG', `Adding ${data.citations.length} citations to response`);
    messageContent += "\n\nCitations:\n";
    data.citations.forEach((citation: string, index: number) => {
      messageContent += `[${index + 1}] ${citation}\n`;
    });
  }

  log('INFO', `Chat completion finished. Response length: ${messageContent.length} characters`);
  return messageContent;
}

/**
 * Fixed Custom HTTP Transport for MCP 
 */
class HTTPStreamTransport {
  private response: http.ServerResponse;
  private requestId: string | number | null = null;

  constructor(response: http.ServerResponse) {
    this.response = response;
  }

  async handleMCPRequest(request: any): Promise<void> {
    this.requestId = request.id;
    log('INFO', `Handling MCP request: ${request.method}`, { id: this.requestId, method: request.method });

    try {
      // Set up proper MCP response headers
      this.response.writeHead(200, {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
      });

      // Process the MCP request
      const { method, params } = request;
      let result;

      if (method === 'initialize') {
        log('INFO', 'Processing initialize request');
        result = {
          protocolVersion: "2024-11-05",
          capabilities: {
            tools: {},
            logging: {},
          },
          serverInfo: {
            name: "perplexity-ask",
            version: "0.1.0"
          }
        };
      } else if (method === 'tools/list') {
        log('INFO', 'Processing tools/list request');
        result = {
          tools: [PERPLEXITY_ASK_TOOL, PERPLEXITY_RESEARCH_TOOL, PERPLEXITY_REASON_TOOL]
        };
      } else if (method === 'tools/call') {
        log('INFO', `Processing tools/call request for tool: ${params?.name}`);
        result = await this.handleToolCall(params);
      } else {
        log('ERROR', `Unknown method: ${method}`);
        throw new Error(`Unknown method: ${method}`);
      }

      // Send the final response
      this.sendResponse(result);
    } catch (error) {
      log('ERROR', 'Error handling MCP request', error);
      this.sendError(error);
    }
  }

  private async handleToolCall(params: any): Promise<any> {
    const { name, arguments: args } = params;
    
    log('INFO', `Executing tool: ${name}`, { arguments: args });
    
    if (!args) {
      throw new Error("No arguments provided");
    }

    switch (name) {
      case "perplexity_ask": {
        if (!Array.isArray(args.messages)) {
          throw new Error("Invalid arguments for perplexity_ask: 'messages' must be an array");
        }
        log('DEBUG', `Calling perplexity_ask with ${args.messages.length} messages`);
        const result = await performChatCompletion(args.messages, "sonar-pro");
        return {
          content: [{ type: "text", text: result }],
          isError: false,
        };
      }
      case "perplexity_research": {
        if (!Array.isArray(args.messages)) {
          throw new Error("Invalid arguments for perplexity_research: 'messages' must be an array");
        }
        log('DEBUG', `Calling perplexity_research with ${args.messages.length} messages`);
        const result = await performChatCompletion(args.messages, "sonar-deep-research");
        return {
          content: [{ type: "text", text: result }],
          isError: false,
        };
      }
      case "perplexity_reason": {
        if (!Array.isArray(args.messages)) {
          throw new Error("Invalid arguments for perplexity_reason: 'messages' must be an array");
        }
        log('DEBUG', `Calling perplexity_reason with ${args.messages.length} messages`);
        const result = await performChatCompletion(args.messages, "sonar-reasoning-pro");
        return {
          content: [{ type: "text", text: result }],
          isError: false,
        };
      }
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  private sendResponse(result: any): void {
    const mcpResponse = {
      jsonrpc: "2.0" as const,
      id: this.requestId || 0,
      result: result
    };
    
    const responseString = JSON.stringify(mcpResponse);
    log('DEBUG', 'Sending MCP response', { response: mcpResponse });
    
    this.response.write(responseString);
    this.response.end();
  }

  private sendError(error: any): void {
    const errorResponse = {
      jsonrpc: "2.0" as const,
      id: this.requestId || 0,
      error: {
        code: -32603,
        message: error instanceof Error ? error.message : String(error),
        data: error
      }
    };
    
    const responseString = JSON.stringify(errorResponse);
    log('ERROR', 'Sending MCP error response', { error: errorResponse });
    
    this.response.write(responseString);
    this.response.end();
  }
}

/**
 * MCP-compliant streaming function for chat completion (non-HTTP streaming)
 * This version doesn't stream in HTTP mode, just returns the complete result
 */
async function performMCPStreamingCompletion(
  messages: Array<{ role: string; content: string }>,
  model: string = "sonar-pro",
  onProgress?: (content: string) => void
): Promise<string> {
  log('INFO', `Starting streaming completion with model: ${model}`);
  
  // For now, we'll use the non-streaming version since MCP HTTP doesn't support streaming properly
  // You could implement streaming later if needed for specific clients
  return await performChatCompletion(messages, model);
}

// Initialize the server with tool metadata and capabilities
const server = new Server(
  {
    name: "example-servers/perplexity-ask",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

/**
 * Registers a handler for listing available tools.
 * When the client requests a list of tools, this handler returns all available Perplexity tools.
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  log('INFO', 'Tools list requested');
  const tools = [PERPLEXITY_ASK_TOOL, PERPLEXITY_RESEARCH_TOOL, PERPLEXITY_REASON_TOOL];
  log('DEBUG', 'Returning tools list', { count: tools.length, tools: tools.map(t => t.name) });
  return { tools };
});

/**
 * Registers a handler for calling a specific tool.
 * Processes requests by validating input and invoking the appropriate tool.
 *
 * @param {object} request - The incoming tool call request.
 * @returns {Promise<object>} The response containing the tool's result or an error.
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  log('INFO', `Tool call received: ${name}`, { arguments: args });
  
  try {
    if (!args) {
      throw new Error("No arguments provided");
    }
    
    switch (name) {
      case "perplexity_ask": {
        if (!Array.isArray(args.messages)) {
          throw new Error("Invalid arguments for perplexity_ask: 'messages' must be an array");
        }
        log('DEBUG', `Executing perplexity_ask with ${args.messages.length} messages`);
        const result = await performChatCompletion(args.messages, "sonar-pro");
        log('INFO', `perplexity_ask completed successfully. Response length: ${result.length}`);
        return {
          content: [{ type: "text", text: result }],
          isError: false,
        };
      }
      case "perplexity_research": {
        if (!Array.isArray(args.messages)) {
          throw new Error("Invalid arguments for perplexity_research: 'messages' must be an array");
        }
        log('DEBUG', `Executing perplexity_research with ${args.messages.length} messages`);
        const result = await performChatCompletion(args.messages, "sonar-deep-research");
        log('INFO', `perplexity_research completed successfully. Response length: ${result.length}`);
        return {
          content: [{ type: "text", text: result }],
          isError: false,
        };
      }
      case "perplexity_reason": {
        if (!Array.isArray(args.messages)) {
          throw new Error("Invalid arguments for perplexity_reason: 'messages' must be an array");
        }
        log('DEBUG', `Executing perplexity_reason with ${args.messages.length} messages`);
        const result = await performChatCompletion(args.messages, "sonar-reasoning-pro");
        log('INFO', `perplexity_reason completed successfully. Response length: ${result.length}`);
        return {
          content: [{ type: "text", text: result }],
          isError: false,
        };
      }
      default:
        log('ERROR', `Unknown tool requested: ${name}`);
        return {
          content: [{ type: "text", text: `Unknown tool: ${name}` }],
          isError: true,
        };
    }
  } catch (error) {
    log('ERROR', `Tool call failed for ${name}`, error);
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error instanceof Error ? error.message : String(error)}`,
        },
      ],
      isError: true,
    };
  }
});

/**
 * Initializes and runs the server using either HTTP or STDIO transport.
 * The transport method is determined by environment variables or command line arguments.
 */
async function runServer() {
  try {
    if (HTTP_MODE) {
      // Run HTTP server
      await runHttpServer();
    } else {
      // Run STDIO server (default)
      await runStdioServer();
    }
  } catch (error) {
    log('ERROR', 'Fatal error running server', error);
    process.exit(1);
  }
}

/**
 * Runs the server using STDIO transport (original functionality).
 */
async function runStdioServer() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  log('INFO', 'Perplexity MCP Server running on stdio with Ask, Research, and Reason tools');
}

/**
 * Runs the server using HTTP transport with MCP streaming capabilities.
 */
async function runHttpServer() {
  const httpServer = http.createServer(async (req, res) => {
    log('DEBUG', `HTTP request received: ${req.method} ${req.url}`);
    
    // Add CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    // Handle preflight requests
    if (req.method === "OPTIONS") {
      log('DEBUG', 'Handling OPTIONS request');
      res.writeHead(200);
      res.end();
      return;
    }

    if (req.method === "GET" && req.url === "/health") {
      log('DEBUG', 'Health check requested');
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "healthy", version: "0.1.0", timestamp: new Date().toISOString() }));
      return;
    }

    if (req.method === "POST" && req.url === "/mcp") {
      log('INFO', 'MCP request received');
      let body = '';
      req.on('data', (chunk) => {
        body += chunk.toString();
      });

      req.on('end', async () => {
        try {
          log('DEBUG', 'Parsing MCP request body', { body });
          const mcpRequest = JSON.parse(body);
          const transport = new HTTPStreamTransport(res);
          await transport.handleMCPRequest(mcpRequest);
        } catch (error) {
          log('ERROR', 'Error processing MCP request', error);
          if (!res.headersSent) {
            res.writeHead(500, { "Content-Type": "application/json" });
          }
          res.end(JSON.stringify({ 
            jsonrpc: "2.0",
            id: null,
            error: {
              code: -32603,
              message: error instanceof Error ? error.message : String(error)
            }
          }));
        }
      });
      return;
    }

    if (req.method === "POST" && req.url === "/tools") {
      log('INFO', 'Legacy tools list requested');
      try {
        const tools = [PERPLEXITY_ASK_TOOL, PERPLEXITY_RESEARCH_TOOL, PERPLEXITY_REASON_TOOL];
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ tools }));
      } catch (error) {
        log('ERROR', 'Error listing tools', error);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Failed to list tools" }));
      }
      return;
    }

    if (req.method === "GET" && req.url === "/") {
      log('DEBUG', 'Info page requested');
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        name: "Perplexity MCP Server",
        version: "0.1.0",
        description: "MCP server for Perplexity API integration with Ask, Research, and Reason tools",
        tools: [
          {
            name: "perplexity_ask",
            description: "General chat and quick responses using sonar-pro model"
          },
          {
            name: "perplexity_research", 
            description: "Deep research queries using sonar-deep-research model"
          },
          {
            name: "perplexity_reason",
            description: "Reasoning and analytical tasks using sonar-reasoning-pro model"
          }
        ],
        endpoints: {
          health: "/health - Health check",
          mcp: "/mcp (POST) - MCP protocol endpoint", 
          tools: "/tools (POST) - List tools (legacy)",
          info: "/ - This information page"
        },
        usage: {
          mcp_example: {
            url: "/mcp",
            method: "POST", 
            body: {
              jsonrpc: "2.0",
              id: 1,
              method: "tools/call",
              params: {
                name: "perplexity_ask",
                arguments: {
                  messages: [
                    { role: "user", content: "What is artificial intelligence?" }
                  ]
                }
              }
            }
          }
        },
        debug: DEBUG ? "Debug mode enabled" : "Debug mode disabled"
      }));
      return;
    }

    // 404 for unknown routes
    log('DEBUG', `404 for unknown route: ${req.method} ${req.url}`);
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Not found" }));
  });

  httpServer.listen(PORT, () => {
    log('INFO', `Perplexity MCP Server running on HTTP port ${PORT}`);
    console.error("Available endpoints:");
    console.error(`  - Health check: http://localhost:${PORT}/health`);
    console.error(`  - MCP endpoint: http://localhost:${PORT}/mcp (POST)`);
    console.error(`  - List tools: http://localhost:${PORT}/tools (POST)`);
    console.error(`  - Info: http://localhost:${PORT}/`);
    console.error(`  - Debug mode: ${DEBUG ? 'enabled' : 'disabled'}`);
  });
}

// Start the server and catch any startup errors
runServer().catch((error) => {
  log('ERROR', 'Fatal error running server', error);
  process.exit(1);
});