# DeepSearch & Research AI Chat

Koyu temalı, **yerel masaüstü** araştırma/sohbet uygulaması. Yanıtları kullanıcının seçtiği yerel Ollama modelinden alır; isteğe bağlı web araştırmasında DuckDuckGo HTML araması yapar, herkese açık sayfaları okur ve kaynakları yanıt bağlamına ekler. Tarayıcı tabanlı bir uygulama değildir.

## Özellikler

- PySide6 ile masaüstü arayüzü; sohbet geçmişi, Markdown tabloları ve Pygments ile vurgulanan kod blokları.
- Ollama `/api/tags` ve akışlı `/api/chat` REST istemcisi. Varsayılan adres **`http://localhost:11435`**; Ayarlar panelinden değiştirilebilir.
- İstek başına web arama anahtarı, DuckDuckGo arama uç noktası, zaman aşımı, sonuç sayısı ve tarama derinliği ayarları.
- Ollama araç çağrısı desteği: `web_search(query)` ve `fetch_web_page(url)`. Araç çağrısı desteklemeyen modellerde de ilk arama ve seçilen ilk sayfalar otomatik olarak modele bağlam olur.
- Canlı arama / sayfa tarama kartları, kullanıcıdan `.txt`, `.md`, `.py`, `.json`, `.csv`, `.log` ekleri ve URL ekleme.
- SQLite sohbet geçmişi ve günlük istek, giriş/çıkış tokenı, arama ve etkin süre sayaçları.
- Ana uygulamayla aynı şifreli kota dosyasını kullanan bağımsız **UsageLimitEditor**.
- GitHub Actions test, Windows `.exe` paketleme ve Artifact yükleme akışı.

## Geliştirme ortamında çalıştırma

Python 3.10 veya daha yenisi ve Windows/macOS/Linux üzerinde Qt gerekir.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m deepsearch
```

Yönetici kota editörünü ayrı çalıştırmak için:

```bash
python -m deepsearch.admin
```

### Ollama

Ollama yerel sunucusunu `11435` portunda çalıştırın veya uygulamanın Ayarlar bölümünde kullandığınız adresi yazın. Örnek (PowerShell):

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama serve
```

Başka bir terminalde Ollama'ya bir model indirin (örnek):

```bash
ollama pull qwen3:8b
```

Uygulamadaki **Yenile** düğmesi kurulu modelleri listeler. Yerel Ollama kurulumunuzun sürüm/model etiketine göre uygun bir model seçin.

### Web araştırması

Varsayılan arama DuckDuckGo'nun HTML uç noktasını kullanır; ayrı API anahtarı istemez. Web araması için uygulamanın çalıştığı bilgisayarın internete erişmesi gerekir. Dış servisler aramayı sınırlayabilir veya sayfa HTML yapısını değiştirebilir; arama başarısız olursa yerel sohbet yine devam eder ve durum kartında hata gösterilir. Yalnızca `http`/`https` adresleri kabul edilir; özel/yerel IP adresleri ve HTTPS'ten HTTP'ye yönlendirmeler engellenir.

## Kota ve yerel veri güvenliği

İlk çalıştırmada kota varsayılanları oluşturulur: **40 istek/gün, 100.000 toplam token/gün, 50 web araması/gün**. Kota editörü bu değerleri sonlu ve pozitif sınırlar içinde değiştirebilir. Geliştirici modu tanılama içindir; **kotaları devre dışı bırakmaz**. Ana uygulama limit dosyasını her yeni istek öncesinde yeniden okur; limit dolduğunda istekleri engeller.

Kota ayarları kullanıcı verisi dizinindeki Fernet ile kimliği doğrulanmış şifreli `limits.enc` dosyasında saklanır; anahtar ayrı tutulur ve POSIX sistemlerinde dosya izinleri daraltılır. Bu, yerel dosyanın okunmasını ve kazara değiştirilmesini zorlaştırır; aynı işletim sistemi hesabını ve çalıştırılabilir dosyaları kontrol eden bir kişiye karşı **aşılamaz bir lisans/kurumsal yetkilendirme sınırı değildir**. Yerel yönetici uygulaması da aynı kullanıcı hesabının erişim alanındadır. Böyle bir tehdit modelinde merkezi bir servis ve işletim sistemi yöneticisi tarafından korunan kimlik bilgileri gerekir.

Token kullanımı Ollama'nın yanıtladığı gerçek `prompt_eval_count` ve `eval_count` değerleriyle kaydedilir. Yeni istekte kalan kota için korumacı bir tahmin ve `num_predict` sınırı uygulanır; Ollama'nın tüm sürümlerinde önceden kesin tokenizer ölçümü bulunmadığından çok küçük bir kota aşımı teknik olarak mümkün olabilir. Kullanım verisi ve konuşma geçmişi yerel SQLite dosyasındadır; uygulama sohbetleri üçüncü taraf bir LLM servisine göndermez. Web araması açıkken arama sorgusu DuckDuckGo'ya iletilir; uzak Ollama adresi seçilirse model mesajları o adrese gönderilir.

Varsayılan veri dizinleri:

- Windows: `%LOCALAPPDATA%\DeepSearchResearchAI`
- macOS: `~/Library/Application Support/DeepSearchResearchAI`
- Linux: `$XDG_DATA_HOME/deepsearch-research-ai` veya `~/.local/share/deepsearch-research-ai`

Testler için `DEEPSEARCH_DATA_DIR` ortam değişkeniyle farklı bir dizin belirlenebilir.

## Test

```bash
python -m pytest
```

Testler yerel/mok verilerle çalışır; Ollama veya internet bağlantısı gerektirmez.

## GitHub Actions / Windows `.exe`

`.github/workflows/build.yml` her push ve `workflow_dispatch` tetiklemesinde önce testleri çalıştırır, ardından Windows üzerinde iki bağımsız tek dosya uygulaması derler:

- `DeepSearch_Chat_App.exe`
- `UsageLimitEditor.exe`

Başarılı derlemede ikisi de 30 günlük GitHub Actions Artifact'ı olarak yüklenir. Taslak GitHub Release, bir `v*` / `release-*` etiketi push edildiğinde veya `workflow_dispatch` ekranında **Create a draft GitHub Release** seçildiğinde oluşturulur. Taslak sürüm yayımlanmadan önce insan tarafından incelenebilir.

## Proje yapısı

```text
src/deepsearch/
  config.py       ayarlar ve kullanıcı veri yolu
  limits.py       şifreli kota deposu
  storage.py      SQLite geçmişi ve sayaçlar
  ollama.py       akışlı Ollama REST istemcisi
  web.py          web araması, SSRF temel koruması ve sayfa çıkarımı
  research.py     araç döngüsü ve kaynak bağlamı
  qt_app.py       ana masaüstü uygulaması
  admin.py        bağımsız kota düzenleyicisi
.github/workflows/build.yml
tests/
```
