import { Widget, WidgetInput, WidgetOutput } from '../types';
import { createSearchDebugLogger } from '../debug';

class WidgetExecutor {
  static widgets = new Map<string, Widget>();

  static register(widget: Widget) {
    this.widgets.set(widget.type, widget);
  }

  static getWidget(type: string): Widget | undefined {
    return this.widgets.get(type);
  }

  static async executeAll(input: WidgetInput): Promise<WidgetOutput[]> {
    const results: WidgetOutput[] = [];
    const log = createSearchDebugLogger(input.sessionId);

    log('widgets:execute:start', { registered: this.widgets.size });

    await Promise.all(
      Array.from(this.widgets.values()).map(async (widget) => {
        try {
          if (widget.shouldExecute(input.classification)) {
            const startTime = Date.now();
            log('widget:execute:start', { type: widget.type });
            const output = await widget.execute(input);
            if (output) {
              results.push(output);
            }
            log('widget:execute:done', {
              type: widget.type,
              output: Boolean(output),
              durationMs: Date.now() - startTime,
            });
          }
        } catch (e) {
          console.log(`Error executing widget ${widget.type}:`, e);
          log('widget:execute:error', {
            type: widget.type,
            error: e instanceof Error ? e.message : String(e),
          });
        }
      }),
    );

    log('widgets:execute:done', { outputs: results.length });
    return results;
  }
}

export default WidgetExecutor;
