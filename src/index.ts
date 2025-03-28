import { config } from "dotenv";
import { Document, VectorStoreIndex } from "llamaindex";
import { OpenAI } from "@llamaindex/openai";
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

    try {
        // Initialize OpenAI
        const openai = new OpenAI();

        // Load PDF document
        const document = await loadPDFDocument("data/sample.pdf");

        // Create vector store index
        const index = await VectorStoreIndex.fromDocuments([document]);

        console.log("PDF document loaded and indexed!");

        // Query engine
        const queryEngine = index.asQueryEngine();

        // Example query
        const response = await queryEngine.query({
            query: "tell me the first line on pdf?",
        });

        console.log("Query Response:", response.toString());
    } catch (error) {
        console.error("Error:", error);
    }
}

main().catch(console.error);
