/**
 * @license Copyright (c) 2014-2023, CKSource Holding sp. z o.o. All rights reserved.
 * For licensing, see LICENSE.md or https://ckeditor.com/legal/ckeditor-oss-license
 */

import { BalloonEditor } from '@ckeditor/ckeditor5-editor-balloon';

import { Bold, Italic } from '@ckeditor/ckeditor5-basic-styles';
import type { EditorConfig } from '@ckeditor/ckeditor5-core';
import { Essentials } from '@ckeditor/ckeditor5-essentials';
import { Link } from '@ckeditor/ckeditor5-link';
import { Paragraph } from '@ckeditor/ckeditor5-paragraph';

// You can read more about extending the build with additional plugins in the "Installing plugins" guide.
// See https://ckeditor.com/docs/ckeditor5/latest/installation/plugins/installing-plugins.html for details.

class Editor extends BalloonEditor {
	public static override builtinPlugins = [
		Bold,
		Essentials,
		Italic,
		Link,
		Paragraph
	];

	public static override defaultConfig: EditorConfig = {
		toolbar: {
			items: [
				'bold',
				'italic',
				'link',
				'|',
				'undo',
				'redo'
			]
		},
		language: 'en'
	};
}

export default Editor;
