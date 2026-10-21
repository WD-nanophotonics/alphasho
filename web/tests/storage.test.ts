import { expect, it } from 'vitest';
import { storageKey } from '../src/storage';
it('keeps legacy production saves while isolating independent Pages previews',()=>{
 expect(storageKey('alphasho.game','/alphasho/')).toBe('alphasho.game');
 expect(storageKey('alphasho.game','/alphasho-nn-preview/')).not.toBe('alphasho.game');
 expect(storageKey('alphasho.language','/alphasho-nn-preview/')).not.toBe(storageKey('alphasho.language','/alphasho/'));
});
