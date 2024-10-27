import { App, Editor, MarkdownView, Modal, Notice, Plugin, PluginSettingTab, Setting } from 'obsidian';
import OpenAI from "openai";

// Define the plugin settings interface
interface MyPluginSettings {
	openAIKey: string;
	selectedModel: string; // Add a setting to store the selected model
}

const DEFAULT_SETTINGS: MyPluginSettings = {
	openAIKey: '',
	selectedModel: 'gpt-4-turbo' // Default model is gpt-4-turbo for cost efficiency
}

export default class MyPlugin extends Plugin {
	settings: MyPluginSettings;
	openai: OpenAI;

	async onload() {
		await this.loadSettings();

		// Initialize OpenAI instance if API key is provided
		if (this.settings.openAIKey) {
			this.openai = new OpenAI({ apiKey: this.settings.openAIKey });
		}

		// Add an icon in the left ribbon
		const ribbonIconEl = this.addRibbonIcon('dice', 'Note AI Plugin', (evt: MouseEvent) => {
			new Notice('Welcome to the Note AI Plugin!');
		});
		ribbonIconEl.addClass('my-plugin-ribbon-class');

		// Add a command to open a chat modal
		this.addCommand({
			id: 'open-chat-modal',
			name: 'Ask a question to your notes',
			callback: () => {
				if (!this.openai) {
					new Notice("Please set your OpenAI API key in the settings.");
					return;
				}
				new ChatModal(this.app, this.openai, this.settings.selectedModel).open();
			}
		});

		// Add a settings tab to configure the API key and model selection
		this.addSettingTab(new RAGSettingTab(this.app, this));
	}

	onunload() {}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings() {
		await this.saveData(this.settings);

		// Update OpenAI instance if API key changes
		if (this.settings.openAIKey) {
			this.openai = new OpenAI({ apiKey: this.settings.openAIKey });
		}
	}
}

// Define the chat modal
class ChatModal extends Modal {
	openai: OpenAI;
	selectedModel: string;

	constructor(app: App, openai: OpenAI, selectedModel: string) {
		super(app);
		this.openai = openai;
		this.selectedModel = selectedModel;
	}

	async onOpen() {
		const { contentEl } = this;
		contentEl.createEl('h2', { text: 'Ask a question based on your notes' });

		const modelSelect = contentEl.createEl('select');
		const models = ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'];

		// Populate the dropdown with available models
		models.forEach(model => {
			const option = modelSelect.createEl('option', { text: model });
			option.value = model;
			if (model === this.selectedModel) option.selected = true;
		});

		// Update the selected model when the user changes the dropdown value
		modelSelect.onchange = (e) => {
			this.selectedModel = (e.target as HTMLSelectElement).value;
		};

		contentEl.appendChild(modelSelect);

		const inputEl = contentEl.createEl('input', { type: 'text', placeholder: 'Type your question here...' });
		const buttonEl = contentEl.createEl('button', { text: 'Submit' });

		buttonEl.onclick = async () => {
			const query = inputEl.value;
			if (!query) {
				new Notice('Please enter a question.');
				return;
			}

			new Notice('Retrieving relevant notes...');
			const notes = await getNotes(this.app);
			const relevantChunks = await getRelevantChunks(query, notes, this.openai);
			const response = await generateAnswer(query, relevantChunks, this.openai, this.selectedModel);

			contentEl.createEl('p', { text: `Answer: ${response}` });
		};
	}

	onClose() {
		const { contentEl } = this;
		contentEl.empty();
	}
}

// Define the settings tab
class RAGSettingTab extends PluginSettingTab {
	plugin: MyPlugin;

	constructor(app: App, plugin: MyPlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const { containerEl } = this;
		containerEl.empty();

		new Setting(containerEl)
			.setName('OpenAI API Key')
			.setDesc('Enter your OpenAI API Key')
			.addText(text => text
				.setPlaceholder('Enter your API key')
				.setValue(this.plugin.settings.openAIKey)
				.onChange(async (value) => {
					this.plugin.settings.openAIKey = value;
					await this.plugin.saveSettings();
				}));

		new Setting(containerEl)
			.setName('Select Model')
			.setDesc('Choose the OpenAI model to use for responses')
			.addDropdown(dropdown => dropdown
				.addOptions({
					'gpt-4': 'GPT-4',
					'gpt-4-turbo': 'GPT-4 Turbo',
					'gpt-3.5-turbo': 'GPT-3.5 Turbo'
				})
				.setValue(this.plugin.settings.selectedModel)
				.onChange(async (value) => {
					this.plugin.settings.selectedModel = value;
					await this.plugin.saveSettings();
				}));
	}
}

// Helper function to retrieve user notes
async function getNotes(app: App): Promise<string[]> {
	const notes: string[] = [];
	const files = app.vault.getMarkdownFiles();

	for (const file of files) {
		const content = await app.vault.read(file);
		notes.push(content);
	}

	return notes;
}

// Helper function to generate embeddings and find relevant chunks
async function getRelevantChunks(query: string, notes: string[], openai: OpenAI): Promise<string[]> {
	const queryEmbedding = await generateEmbeddings(query, openai);
	const embeddings = await Promise.all(notes.map(note => generateEmbeddings(note, openai)));

	// Calculate similarity scores
	const scores = embeddings.map(embedding => similarity(queryEmbedding, embedding));

	// Select top chunks based on similarity
	const topChunks = notes
		.map((note, index) => ({ note, score: scores[index] }))
		.sort((a, b) => b.score - a.score)
		.slice(0, 5) // Top 5 chunks
		.map(chunk => chunk.note);

	return topChunks;
}

// Helper function to generate embeddings
async function generateEmbeddings(text: string, openai: OpenAI): Promise<number[]> {
	const response = await openai.embeddings.create({
		model: "text-embedding-ada-002",
		input: text,
	});
	return response.data[0].embedding;
}

// Helper function to generate an answer using OpenAI API
async function generateAnswer(query: string, context: string[], openai: OpenAI, model: string): Promise<string> {
	const prompt = `Context:\n${context.join("\n")}\n\nQuestion: ${query}\nAnswer:`;

	const response = await openai.chat.completions.create({
		model: model,
		messages: [{ role: "user", content: prompt }],
	});

	return response.choices[0].message.content ?? "No response generated";
}

// Simple similarity function (e.g., cosine similarity)
function similarity(vec1: number[], vec2: number[]): number {
	const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
	const magnitude1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
	const magnitude2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
	return dotProduct / (magnitude1 * magnitude2);
}
