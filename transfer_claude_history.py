#!/usr/bin/env python3
"""
Скрипт для переноса истории чата из Claude в Cursor.

Поддерживает несколько форматов:
1. JSON экспорт из Claude (если доступен)
2. Текстовый формат (копирование сообщений)
3. Markdown формат

Использование:
    python transfer_claude_history.py --input claude_history.json --output cursor_history.md
    python transfer_claude_history.py --input claude_chat.txt --format text --output cursor_history.md
"""

import json
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class ClaudeHistoryConverter:
    """Конвертер истории чата из Claude в формат для Cursor"""
    
    def __init__(self):
        self.conversations = []
    
    def parse_json(self, json_data: Dict[str, Any]) -> List[Dict]:
        """Парсит JSON формат истории Claude"""
        messages = []
        
        # Различные возможные структуры JSON экспорта Claude
        if isinstance(json_data, dict):
            # Вариант 1: Прямой массив сообщений
            if 'messages' in json_data:
                messages = json_data['messages']
            # Вариант 2: Массив разговоров
            elif 'conversations' in json_data:
                for conv in json_data['conversations']:
                    if 'messages' in conv:
                        messages.extend(conv['messages'])
            # Вариант 3: Прямой объект с полями
            elif 'content' in json_data:
                messages = [json_data]
        
        elif isinstance(json_data, list):
            messages = json_data
        
        result = []
        for msg in messages:
            role = msg.get('role', msg.get('from', 'unknown'))
            content = msg.get('content', msg.get('text', ''))
            
            # Нормализация ролей
            if role.lower() in ['user', 'human', 'you']:
                role = 'user'
            elif role.lower() in ['assistant', 'claude', 'ai', 'bot']:
                role = 'assistant'
            
            result.append({
                'role': role,
                'content': content,
                'timestamp': msg.get('timestamp', msg.get('created_at', None))
            })
        
        return result
    
    def parse_text(self, text: str) -> List[Dict]:
        """Парсит текстовый формат (копирование из чата)"""
        messages = []
        lines = text.split('\n')
        
        current_role = None
        current_content = []
        
        # Паттерны для определения ролей
        user_patterns = [
            re.compile(r'^(Вы|You|User|Human):', re.IGNORECASE),
            re.compile(r'^>', re.MULTILINE)
        ]
        
        assistant_patterns = [
            re.compile(r'^(Claude|Assistant|AI|Бот):', re.IGNORECASE),
            re.compile(r'^Assistant:', re.IGNORECASE)
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Проверка на начало нового сообщения
            is_user = any(pattern.match(line) for pattern in user_patterns)
            is_assistant = any(pattern.match(line) for pattern in assistant_patterns)
            
            if is_user or is_assistant:
                # Сохраняем предыдущее сообщение
                if current_role and current_content:
                    messages.append({
                        'role': current_role,
                        'content': '\n'.join(current_content)
                    })
                
                # Начинаем новое сообщение
                current_role = 'user' if is_user else 'assistant'
                # Убираем префикс роли
                content = re.sub(r'^(Вы|You|User|Human|Claude|Assistant|AI|Бот):\s*', '', line, flags=re.IGNORECASE)
                content = re.sub(r'^>\s*', '', content)
                current_content = [content] if content else []
            else:
                # Продолжение текущего сообщения
                if current_role:
                    current_content.append(line)
        
        # Сохраняем последнее сообщение
        if current_role and current_content:
            messages.append({
                'role': current_role,
                'content': '\n'.join(current_content)
            })
        
        return messages
    
    def to_markdown(self, messages: List[Dict]) -> str:
        """Конвертирует сообщения в Markdown формат для Cursor"""
        md_lines = []
        md_lines.append("# История чата из Claude\n")
        md_lines.append(f"*Перенесено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        md_lines.append("---\n\n")
        
        for i, msg in enumerate(messages, 1):
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                md_lines.append(f"## Сообщение {i}: Пользователь\n\n")
                md_lines.append(f"{content}\n\n")
            elif role == 'assistant':
                md_lines.append(f"## Сообщение {i}: Claude\n\n")
                md_lines.append(f"{content}\n\n")
            
            md_lines.append("---\n\n")
        
        return '\n'.join(md_lines)
    
    def to_cursor_format(self, messages: List[Dict]) -> str:
        """Конвертирует в формат, удобный для Cursor"""
        cursor_lines = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                cursor_lines.append(f"**User:**\n{content}\n\n")
            elif role == 'assistant':
                cursor_lines.append(f"**Assistant:**\n{content}\n\n")
        
        return '\n'.join(cursor_lines)
    
    def convert(self, input_file: str, output_file: str, input_format: str = 'auto'):
        """Основной метод конвертации"""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_file}")
        
        # Определяем формат автоматически
        if input_format == 'auto':
            if input_path.suffix.lower() == '.json':
                input_format = 'json'
            else:
                input_format = 'text'
        
        # Читаем входной файл
        with open(input_path, 'r', encoding='utf-8') as f:
            if input_format == 'json':
                data = json.load(f)
                messages = self.parse_json(data)
            else:
                text = f.read()
                messages = self.parse_text(text)
        
        # Конвертируем в формат для Cursor
        output_path = Path(output_file)
        if output_path.suffix.lower() == '.md':
            output_content = self.to_markdown(messages)
        else:
            output_content = self.to_cursor_format(messages)
        
        # Сохраняем результат
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"✓ Успешно конвертировано {len(messages)} сообщений")
        print(f"✓ Результат сохранен в: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Перенос истории чата из Claude в Cursor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Конвертация из JSON:
   python transfer_claude_history.py --input claude_export.json --output cursor_history.md

2. Конвертация из текстового файла:
   python transfer_claude_history.py --input claude_chat.txt --format text --output cursor_history.md

3. Автоматическое определение формата:
   python transfer_claude_history.py --input history.txt --output result.md
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Входной файл с историей Claude (JSON или текст)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Выходной файл для сохранения результата'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['auto', 'json', 'text'],
        default='auto',
        help='Формат входного файла (по умолчанию: auto)'
    )
    
    args = parser.parse_args()
    
    converter = ClaudeHistoryConverter()
    try:
        converter.convert(args.input, args.output, args.format)
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
