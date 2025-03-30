import { config } from "dotenv";
import { Document, VectorStoreIndex, storageContextFromDefaults } from "llamaindex";
import { OpenAI } from "@llamaindex/openai";
import { PGVectorStore } from "@llamaindex/postgres";
import { Client } from "pg"; // Import PostgreSQL client
import * as fs from "fs";
import pdf from "pdf-parse";

// Load environment variables from .env file
config();

async function loadPDFDocument(filePath: string): Promise<Document> {
    const pdfBuffer = fs.readFileSync(filePath);
    const pdfData = await pdf(pdfBuffer);

    return new Document({
        text: pdfData.text,
        metadata: { fileName: filePath },
    });
}

async function main() {
    if (!process.env.OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY environment variable is not set");
    }

    if (!process.env.DATABASE_URL) {
        throw new Error("DATABASE_URL environment variable is not set");
    }

    // Initialize PostgreSQL client
    const client = new Client({
        connectionString: process.env.DATABASE_URL,
    });

    try {
        // Connect to the database
        await client.connect();

        // Initialize PostgreSQL vector store
        const postgresStore = new PGVectorStore({
            client, // Pass the connected client
            tableName: "document_vectors", // This table will be created automatically
        });

        // Create storage context with Postgres
        const storageContext = await storageContextFromDefaults({
            vectorStore: postgresStore,
        });

        // Load PDF document
        const document = await loadPDFDocument("data/sample.pdf");

        // Create vector store index
        const index = await VectorStoreIndex.fromDocuments([document], {
            storageContext,
        });

        console.log("✅ PDF document loaded and indexed in PostgreSQL!");

        // Query engine
        const queryEngine = index.asQueryEngine();

        // Example query
        const response = await queryEngine.query({
            query: "What is this document about?",
        });

        console.log("🔍 Query Response:", response.toString());
    } catch (error) {
        console.error("❌ Error:", error);
    } finally {
        // Close the database connection
        await client.end();
    }
}

main().catch(console.error);
