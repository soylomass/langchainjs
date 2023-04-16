import { BaseChatMemory } from "./chat_memory.js";
import { ConversationSummaryMemory } from "./summary.js";
import { BaseChatMessage } from "../schema/index.js";
import { MemoryVariables, InputValues, OutputValues, getBufferString } from "./base.js";

export class ConversationSummaryBufferMemory extends ConversationSummaryMemory {
  maxTokenLimit = 2000;
  movingSummaryBuffer = "";
  memoryKey = "history";

  get memoryVariables(): string[] {
    return [this.memoryKey];
  }

  async loadMemoryVariables(_: InputValues): Promise<MemoryVariables> {
    let buffer = this.chatHistory.messages;
    if (this.movingSummaryBuffer !== "") {
      const firstMessages: BaseChatMessage[] = [
        new this.summaryChatMessageClass(this.movingSummaryBuffer),
      ];
      buffer = firstMessages.concat(buffer);
    }
    const finalBuffer = this.returnMessages
      ? buffer
      : getBufferString(buffer, this.humanPrefix, this.aiPrefix);
    return { [this.memoryKey]: finalBuffer };
  }

  async saveContext(
    inputValues: InputValues,
    outputValues: OutputValues
  ): Promise<void> {
    await super.saveContext(inputValues, outputValues);
    // Prune buffer if it exceeds max token limit
    const buffer = this.chatHistory.messages;
    // get all messages text concatenated
    let currBufferLength = await this.llm.getNumTokens(getBufferString(buffer));
    if (currBufferLength > this.maxTokenLimit) {
      const prunedMemory: BaseChatMessage[] = [];
      while (currBufferLength > this.maxTokenLimit) {
        prunedMemory.push(buffer.shift() as BaseChatMessage);
        currBufferLength = await this.llm.getNumTokens(getBufferString(buffer));
      }
      this.movingSummaryBuffer = await this.predictNewSummary(
        prunedMemory,
        this.movingSummaryBuffer
      );
    }
  }

  async clear(): Promise<void> {
    await super.clear();
    this.movingSummaryBuffer = "";
  }
}